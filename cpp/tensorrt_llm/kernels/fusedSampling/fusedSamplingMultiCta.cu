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

#include "fusedSamplingKernelsCommon.cuh"

namespace tensorrt_llm
{
namespace kernels
{
namespace fusedSampling
{

namespace
{

//! Several CTAs independently reduce contiguous vocabulary slices and leave their
//! online-softmax partials at the start of the output row.
template <typename T>
__global__ void smallBatchSplitStatsKernel(FusedSamplingParams params)
{
    int const row = blockIdx.x / kSmallBatchSplits;
    int const split = blockIdx.x % kSmallBatchSplits;
    int const tid = threadIdx.x;
    RowParams const rp = loadRowParams(params, row);

    int const vocabSize = params.vocabSize;
    int const begin = static_cast<int>(static_cast<long long>(vocabSize) * split / kSmallBatchSplits);
    int const end = static_cast<int>(static_cast<long long>(vocabSize) * (split + 1) / kSmallBatchSplits);
    T const* rowLogits = static_cast<T const*>(params.logits) + static_cast<size_t>(row) * vocabSize;

    OnlineSoftmax local{-FLT_MAX, 0.0f, begin};
    for (int i = begin + tid; i < end; i += blockDim.x)
    {
        local = combineOnlineSoftmax(local, OnlineSoftmax{__fmul_rn(toFloat(rowLogits[i]), rp.tempInv), 1.0f, i});
    }

    using BlockReduceOnline = cub::BlockReduce<OnlineSoftmax, kSmallBatchSplitBlock>;
    __shared__ typename BlockReduceOnline::TempStorage temp;
    OnlineSoftmax const partial = BlockReduceOnline(temp).Reduce(local, OnlineSoftmaxOp());
    if (tid == 0)
    {
        float* rowProbs = params.outputProbs + static_cast<size_t>(row) * vocabSize;
        reinterpret_cast<OnlineSoftmax*>(rowProbs)[split] = partial;
    }
}

//! Reuse the split-vocabulary grid to write neutral probabilities or build pure-top-p's
//! first radix histogram. Other filters leave their merged statistics for the final CTA.
template <typename T, bool NEED_TOKENS>
__global__ void smallBatchSplitOutputKernel(FusedSamplingParams params)
{
    int const row = blockIdx.x / kSmallBatchSplits;
    int const split = blockIdx.x % kSmallBatchSplits;
    int const tid = threadIdx.x;
    int const vocabSize = params.vocabSize;
    RowParams const rp = loadRowParams(params, row);

    T const* rowLogits = static_cast<T const*>(params.logits) + static_cast<size_t>(row) * vocabSize;
    float* rowProbs = params.outputProbs + static_cast<size_t>(row) * vocabSize;
    int const splitBegin = static_cast<int>(static_cast<long long>(vocabSize) * split / kSmallBatchSplits);
    int const splitEnd = static_cast<int>(static_cast<long long>(vocabSize) * (split + 1) / kSmallBatchSplits);
    __shared__ OnlineSoftmax sRowStats;
    if (tid == 0)
    {
        OnlineSoftmax rowStats = reinterpret_cast<OnlineSoftmax const*>(rowProbs)[0];
        for (int other = 1; other < kSmallBatchSplits; ++other)
        {
            rowStats = combineOnlineSoftmax(rowStats, reinterpret_cast<OnlineSoftmax const*>(rowProbs)[other]);
        }
        sRowStats = rowStats;
        if (split == 0)
        {
            *reinterpret_cast<OnlineSoftmax*>(rowProbs + kMergedStatsOffset) = rowStats;
        }
    }
    __syncthreads();
    OnlineSoftmax const rowStats = sRowStats;
    if (rp.needTopK || rp.needMinP)
    {
        return;
    }

    // The first radix pass is dominated by contended shared atomics. Spread it across
    // the same slice grid as the online-softmax pass, then leave one folded histogram per
    // slice in the output buffer for the row CTA to merge and continue descending.
    __shared__ int sCount[kRadixBuckets * kHistCopies];
    __shared__ float sMass[kRadixBuckets * kHistCopies];
    if (rp.needTopP)
    {
        for (int b = tid; b < kRadixBuckets * kHistCopies; b += blockDim.x)
        {
            sCount[b] = 0;
            sMass[b] = 0.0f;
        }
        __syncthreads();

        int const copy = (tid >> 5) & (kHistCopies - 1);
        int* const myCount = sCount + copy * kRadixBuckets;
        float* const myMass = sMass + copy * kRadixBuckets;
        for (int i = splitBegin + tid; i < splitEnd; i += blockDim.x)
        {
            float const w = weightOf(rowLogits, i, rp.tempInv, rowStats.max);
            uint32_t const digit = __float_as_uint(w) >> (32 - kRadixBits);
            atomicAdd(&myCount[digit], 1);
            atomicAdd(&myMass[digit], w);
        }
        __syncthreads();

        float* const splitHistogram = rowProbs + kSplitHistogramOffset + split * kSplitHistogramStride;
        for (int b = tid; b < kRadixBuckets; b += blockDim.x)
        {
            int totalCount = sCount[b];
            float totalMass = sMass[b];
#pragma unroll
            for (int c = 1; c < kHistCopies; ++c)
            {
                totalCount += sCount[c * kRadixBuckets + b];
                totalMass += sMass[c * kRadixBuckets + b];
            }
            reinterpret_cast<int*>(splitHistogram)[b] = totalCount;
            splitHistogram[kRadixBuckets + b] = totalMass;
        }
        return;
    }

    float const scale = 1.0f / rowStats.sum;
    int const begin = splitBegin > kSmallBatchWorkspaceFloats ? splitBegin : kSmallBatchWorkspaceFloats;

    int const vecBegin = (begin + 3) / 4;
    int const vecEnd = splitEnd / 4;
    float localProbMass = 0.0f;
    int const prologueEnd = vecBegin * 4 < splitEnd ? vecBegin * 4 : splitEnd;
    for (int i = begin + tid; i < prologueEnd; i += blockDim.x)
    {
        float const p = weightOf(rowLogits, i, rp.tempInv, rowStats.max) * scale;
        rowProbs[i] = p;
        if constexpr (NEED_TOKENS)
        {
            localProbMass += p;
        }
    }
    for (int v = vecBegin + tid; v < vecEnd; v += blockDim.x)
    {
        float4 out;
        float* elems = reinterpret_cast<float*>(&out);
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            elems[j] = weightOf(rowLogits, v * 4 + j, rp.tempInv, rowStats.max) * scale;
            if constexpr (NEED_TOKENS)
            {
                localProbMass += elems[j];
            }
        }
        reinterpret_cast<float4*>(rowProbs)[v] = out;
    }
    int const tailBegin = begin > vecEnd * 4 ? begin : vecEnd * 4;
    for (int i = tailBegin + tid; i < splitEnd; i += blockDim.x)
    {
        float const p = weightOf(rowLogits, i, rp.tempInv, rowStats.max) * scale;
        rowProbs[i] = p;
        if constexpr (NEED_TOKENS)
        {
            localProbMass += p;
        }
    }

    if constexpr (NEED_TOKENS)
    {
        using BlockReduceF = cub::BlockReduce<float, kSmallBatchSplitBlock>;
        __shared__ typename BlockReduceF::TempStorage temp;
        float const splitMass = BlockReduceF(temp).Sum(localProbMass);
        if (tid == 0)
        {
            rowProbs[kSplitMassOffset + split] = splitMass;
        }
    }
}

//! Finish split-owned rows, or continue the normal fused body from the precomputed
//! statistics. Mixing both in one grid keeps heterogeneous rows concurrent.
template <typename T, bool NEED_TOKENS>
__global__ void smallBatchSplitFinalizeKernel(FusedSamplingParams params)
{
    int const row = blockIdx.x;
    int const tid = threadIdx.x;
    int const vocabSize = params.vocabSize;
    RowParams const rp = loadRowParams(params, row);
    __shared__ FusedSamplingShared<kWideBlock> shared;
    if (rp.needTopK || rp.needMinP)
    {
        fusedSamplingBody<T, kWideBlock, NEED_TOKENS, true, true>(params, shared);
        return;
    }

    T const* rowLogits = static_cast<T const*>(params.logits) + static_cast<size_t>(row) * vocabSize;
    float* rowProbs = params.outputProbs + static_cast<size_t>(row) * vocabSize;
    using BlockReduceF = cub::BlockReduce<float, kWideBlock>;
    using BlockScanF = cub::BlockScan<float, kWideBlock>;
    auto& temp = shared.temp;
    int* const sCount = shared.count;
    float* const sMass = shared.mass;
    float* const sCand = shared.cand;
    int& sCandCount = shared.candCount;
    int& sBucketCount = shared.bucketCount;
    int& sChosen = shared.chosen;
    long long& sCountHi = shared.countHi;
    float& sMassHi = shared.massHi;
    bool& sFired = shared.fired;
    float& sMaxScaled = shared.maxScaled;
    float& sTotalMass = shared.totalMass;
    float& sKeptMass = shared.keptMass;
    float& sTarget = shared.target;
    int& sArgMax = shared.argmax;
    int& sToken = shared.token;

    if (tid == 0)
    {
        OnlineSoftmax const rowStats = *reinterpret_cast<OnlineSoftmax const*>(rowProbs + kMergedStatsOffset);
        sMaxScaled = rowStats.max;
        sTotalMass = rowStats.sum;
        sArgMax = rowStats.argmax;
        if constexpr (NEED_TOKENS)
        {
            if (!rp.needTopP)
            {
                for (int split = 0; split < kSmallBatchSplits; ++split)
                {
                    sMass[split] = rowProbs[kSplitMassOffset + split];
                }
            }
        }
    }
    __syncthreads();
    float const maxScaled = sMaxScaled;
    float const totalMass = sTotalMass;
    int const argmax = sArgMax;

    if (rp.needTopP)
    {
        // Merge the per-slice first-pass histograms into the layout findThreshold expects:
        // copy 0 is scanned in place, while copy 1 preserves each bucket's raw count.
        for (int b = tid; b < kRadixBuckets; b += blockDim.x)
        {
            int totalCount = 0;
            float totalMass = 0.0f;
#pragma unroll
            for (int split = 0; split < kSmallBatchSplits; ++split)
            {
                float const* splitHistogram = rowProbs + kSplitHistogramOffset + split * kSplitHistogramStride;
                totalCount += reinterpret_cast<int const*>(splitHistogram)[b];
                totalMass += splitHistogram[kRadixBuckets + b];
            }
            sCount[b] = totalCount;
            sCount[kRadixBuckets + b] = totalCount;
            sMass[b] = totalMass;
            sMass[kRadixBuckets + b] = totalMass;
        }
        __syncthreads();

        float threshold = findThreshold<T>(rowLogits, vocabSize, rp.tempInv, maxScaled, 0.0f, 0, rp.topP * totalMass,
            /*byCount=*/false, sCount, sMass, sCand, &sCandCount, &sBucketCount, &sChosen, &sCountHi, &sMassHi, &sFired,
            /*firstHistogramReady=*/true);

        float localKept = 0.0f;
        forEachLogit(rowLogits, vocabSize,
            [&](int, float logit)
            {
                float const w = __expf(__fmul_rn(logit, rp.tempInv) - maxScaled);
                if (w >= threshold)
                {
                    localKept += w;
                }
            });
        float const blockKept = BlockReduceF(temp.reduceF).Sum(localKept);
        if (tid == 0)
        {
            sKeptMass = blockKept;
        }
        __syncthreads();

        float keptMass = sKeptMass;
        if (!(keptMass > 0.0f))
        {
            threshold = 0.0f;
            keptMass = totalMass;
            localKept = 0.0f;
            forEachLogit(rowLogits, vocabSize,
                [&](int, float logit) { localKept += __expf(__fmul_rn(logit, rp.tempInv) - maxScaled); });
        }

        float const outputScale = 1.0f / keptMass;
        int const probVecCount = vocabSize / 4;
        bool const probsAligned = (reinterpret_cast<uintptr_t>(rowProbs) % kVecBytes) == 0;
        if (probsAligned)
        {
            for (int v = tid; v < probVecCount; v += blockDim.x)
            {
                float4 out;
                float* elems = reinterpret_cast<float*>(&out);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    float const w = weightOf(rowLogits, v * 4 + j, rp.tempInv, maxScaled);
                    elems[j] = w >= threshold ? w * outputScale : 0.0f;
                }
                reinterpret_cast<float4*>(rowProbs)[v] = out;
            }
            for (int i = probVecCount * 4 + tid; i < vocabSize; i += blockDim.x)
            {
                float const w = weightOf(rowLogits, i, rp.tempInv, maxScaled);
                rowProbs[i] = w >= threshold ? w * outputScale : 0.0f;
            }
        }
        else
        {
            for (int i = tid; i < vocabSize; i += blockDim.x)
            {
                float const w = weightOf(rowLogits, i, rp.tempInv, maxScaled);
                rowProbs[i] = w >= threshold ? w * outputScale : 0.0f;
            }
        }

        if constexpr (NEED_TOKENS)
        {
            if (tid == 0)
            {
                int const rngIdx = params.perRowRng ? row : 0;
                uint64_t const seed = params.seed != nullptr ? params.seed[rngIdx] : 0ull;
                uint64_t const offset = params.offset != nullptr ? params.offset[rngIdx] : 0ull;
                curandStatePhilox4_32_10_t state;
                curand_init(seed, static_cast<uint64_t>(row), offset, &state);
                sTarget = curand_uniform(&state) * keptMass;
                sToken = -1;
            }
            __syncthreads();

            float base = 0.0f;
            BlockScanF(temp.scanF).ExclusiveSum(localKept, base);
            __syncthreads();
            if (sTarget >= base && sTarget < base + localKept)
            {
                float running = base;
                bool found = false;
                forEachLogit(rowLogits, vocabSize,
                    [&](int i, float logit)
                    {
                        if (found)
                        {
                            return;
                        }
                        float const w = __expf(__fmul_rn(logit, rp.tempInv) - maxScaled);
                        if (w < threshold)
                        {
                            return;
                        }
                        running += w;
                        if (running > sTarget)
                        {
                            sToken = i;
                            found = true;
                        }
                    });
            }
            __syncthreads();
            if (tid == 0)
            {
                params.outputTokens[row] = sToken >= 0 ? sToken : argmax;
            }
        }
        return;
    }

    float const scale = 1.0f / totalMass;
    if constexpr (NEED_TOKENS)
    {
        if (tid == 0)
        {
            float headMass = 0.0f;
            for (int i = 0; i < kSmallBatchWorkspaceFloats; ++i)
            {
                float const p = weightOf(rowLogits, i, rp.tempInv, maxScaled) * scale;
                rowProbs[i] = p;
                headMass += p;
            }
            sMass[0] += headMass;

            float totalProbMass = 0.0f;
            for (int split = 0; split < kSmallBatchSplits; ++split)
            {
                totalProbMass += sMass[split];
            }
            int const rngIdx = params.perRowRng ? row : 0;
            uint64_t const seed = params.seed != nullptr ? params.seed[rngIdx] : 0ull;
            uint64_t const offset = params.offset != nullptr ? params.offset[rngIdx] : 0ull;
            curandStatePhilox4_32_10_t state;
            curand_init(seed, static_cast<uint64_t>(row), offset, &state);
            float const target = curand_uniform(&state) * totalProbMass;

            float massBefore = 0.0f;
            sChosen = kSmallBatchSplits - 1;
            for (int split = 0; split < kSmallBatchSplits; ++split)
            {
                if (target < massBefore + sMass[split])
                {
                    sChosen = split;
                    break;
                }
                massBefore += sMass[split];
            }
            sTarget = target - massBefore;
            sToken = -1;
        }
    }
    else
    {
        for (int i = tid; i < kSmallBatchWorkspaceFloats; i += blockDim.x)
        {
            rowProbs[i] = weightOf(rowLogits, i, rp.tempInv, maxScaled) * scale;
        }
    }

    if constexpr (NEED_TOKENS)
    {
        __syncthreads();
        int const chosenSplit = sChosen;
        int const splitBegin = static_cast<int>(static_cast<long long>(vocabSize) * chosenSplit / kSmallBatchSplits);
        int const splitEnd
            = static_cast<int>(static_cast<long long>(vocabSize) * (chosenSplit + 1) / kSmallBatchSplits);
        float localProbMass = 0.0f;
        for (int i = splitBegin + tid; i < splitEnd; i += blockDim.x)
        {
            localProbMass += rowProbs[i];
        }
        float base = 0.0f;
        BlockScanF(temp.scanF).ExclusiveSum(localProbMass, base);
        __syncthreads();

        float const target = sTarget;
        if (target >= base && target < base + localProbMass)
        {
            float running = base;
            bool found = false;
            for (int i = splitBegin + tid; i < splitEnd; i += blockDim.x)
            {
                if (found)
                {
                    continue;
                }
                running += rowProbs[i];
                if (running > target)
                {
                    sToken = i;
                    found = true;
                }
            }
        }
        __syncthreads();
        if (tid == 0)
        {
            params.outputTokens[row] = sToken >= 0 ? sToken : argmax;
        }
    }
}

} // namespace

template <typename T>
void launchFusedSamplingMultiCta(FusedSamplingParams const& params, cudaStream_t stream)
{
    dim3 const statsGrid(params.numRows * kSmallBatchSplits);
    dim3 const statsBlock(kSmallBatchSplitBlock);
    smallBatchSplitStatsKernel<T><<<statsGrid, statsBlock, 0, stream>>>(params);

    dim3 const rowGrid(params.numRows);
    dim3 const finishBlock(kWideBlock);
    if (params.outputTokens != nullptr)
    {
        smallBatchSplitOutputKernel<T, true><<<statsGrid, statsBlock, 0, stream>>>(params);
        smallBatchSplitFinalizeKernel<T, true><<<rowGrid, finishBlock, 0, stream>>>(params);
    }
    else
    {
        smallBatchSplitOutputKernel<T, false><<<statsGrid, statsBlock, 0, stream>>>(params);
        smallBatchSplitFinalizeKernel<T, false><<<rowGrid, finishBlock, 0, stream>>>(params);
    }
}

template void launchFusedSamplingMultiCta<float>(FusedSamplingParams const&, cudaStream_t);
template void launchFusedSamplingMultiCta<__half>(FusedSamplingParams const&, cudaStream_t);
template void launchFusedSamplingMultiCta<__nv_bfloat16>(FusedSamplingParams const&, cudaStream_t);

} // namespace fusedSampling
} // namespace kernels
} // namespace tensorrt_llm
