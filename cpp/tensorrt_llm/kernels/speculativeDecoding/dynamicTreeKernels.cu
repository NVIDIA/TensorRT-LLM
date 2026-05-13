/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
 * Portions Copyright (c) 2025 by SGLang team (original implementation).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

#include "dynamicTreeKernels.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/common/vec_dtypes.cuh"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <torch/extension.h>
TRTLLM_NAMESPACE_BEGIN

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::runtime;

namespace kernels::speculative_decoding
{

// ---------------------------------------------------------------------------
// Two-stage top-k / top-p masking kernels
// Mirrors the approach in invokeBatchTopKSampling (samplingTopKKernels.cu),
// but outputs a masked logits tensor instead of sampling a token.
// ---------------------------------------------------------------------------

// Stage 1: Parallel top-k reduction across BLOCKS_PER_BEAM_ blocks per row.
// Each block handles (vocabSize / BLOCKS_PER_BEAM_) elements and finds its
// local top-k, writing (global_index, logit_value) pairs into the tmp buffers.
template <typename T, int32_t BLOCK_SIZE_, int32_t BLOCKS_PER_BEAM_>
__global__ void topKProbStage1(T const* __restrict__ logits, T* tmpLogProbs, int32_t* topKTmpIdBuf, T* topKTmpValBuf,
    int32_t maxTopK, int32_t const* topKs, int32_t vocabSize)
{
    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;

    auto const tid = static_cast<int32_t>(threadIdx.x);
    auto const bid = static_cast<int32_t>(blockIdx.x);
    auto const rowId = bid / BLOCKS_PER_BEAM_;
    auto const blockLane = bid % BLOCKS_PER_BEAM_; // chunk index within the row

    auto const k = (topKs != nullptr) ? topKs[rowId] : maxTopK;

    bool const IS_FP16 = std::is_same<T, half>::value;
    T const MAX_T_VAL = IS_FP16 ? HALF_FLT_MAX : FLT_MAX;

    // Base offset into the flat (nRows * vocabSize) logits array for this row.
    auto const rowOffset = rowId * vocabSize;
    // Base offset into the tmp buffers for this (row, blockLane).
    auto const tmpIdxBase = rowId * BLOCKS_PER_BEAM_ * maxTopK + blockLane * k;

    // Copy this block's chunk of logits into tmpLogProbs scratch space.
    for (auto elemId = tid + blockLane * BLOCK_SIZE_; elemId < vocabSize; elemId += BLOCK_SIZE_ * BLOCKS_PER_BEAM_)
    {
        tmpLogProbs[rowOffset + elemId] = logits[rowOffset + elemId];
    }
    __syncthreads();

    // Iteratively find the top-k values via max-reduction, zeroing each found max.
    TopK_2<T> partial;
    for (int32_t ite = 0; ite < k; ite++)
    {
        partial.init();
        for (auto elemId = tid + blockLane * BLOCK_SIZE_; elemId < vocabSize; elemId += BLOCK_SIZE_ * BLOCKS_PER_BEAM_)
        {
            partial.insert(tmpLogProbs[rowOffset + elemId], rowOffset + elemId);
        }

        TopK_2<T> total = BlockReduce(tempStorage).Reduce(partial, reduce_topk_op_2<T>);

        if (tid == 0)
        {
            topKTmpIdBuf[tmpIdxBase + ite] = total.p;  // global index (rowOffset + vocabIdx)
            topKTmpValBuf[tmpIdxBase + ite] = total.u; // logit value
            if (total.p >= 0)
            {
                tmpLogProbs[total.p] = -MAX_T_VAL; // zero out so next iteration finds next-best
            }
        }
        __syncthreads();
    }
}

// Stage 2: Merge BLOCKS_PER_BEAM_ * k candidates per row, apply optional top-p,
// then scatter selected logit values back to an output logits tensor (all other
// positions are set to -inf so that a subsequent softmax produces 0 probability).
template <typename T, int32_t BLOCK_SIZE_, int32_t BLOCKS_PER_BEAM_>
__global__ void topKProbStage2ForLogits(int32_t const* __restrict__ topKTmpIdBuf, T* topKTmpValBuf, float* outputLogits,
    int32_t maxTopK, int32_t const* topKs, float const* topPs, int32_t vocabSize)
{
    bool const IS_FP16 = std::is_same<T, half>::value;
    T const MAX_T_VAL = IS_FP16 ? HALF_FLT_MAX : FLT_MAX;

    auto const tid = static_cast<int32_t>(threadIdx.x);
    auto const rowId = static_cast<int32_t>(blockIdx.x);

    auto const k = (topKs != nullptr) ? topKs[rowId] : maxTopK;
    // size: number of valid candidates written by Stage 1 for this row.
    // stride: row pitch in the tmp buffers (same as in invokeBatchTopKSampling).
    auto const size = k * BLOCKS_PER_BEAM_;
    auto const stride = maxTopK * BLOCKS_PER_BEAM_;

    typedef cub::BlockReduce<TopK_2<float>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    extern __shared__ char sharedArray[];
    // Shared layout: sId[maxTopK] | sVal2[maxTopK]
    auto* sId = reinterpret_cast<int32_t*>(sharedArray);
    auto* sVal2 = reinterpret_cast<float*>(sId + maxTopK);

    // Pointer to this row's candidates in the tmp value buffer (modified in-place during reduction).
    T* sVal = topKTmpValBuf + rowId * stride;

    // Step 1: Initialize output row to -inf (all threads cooperate for bandwidth).
    float* outRow = outputLogits + rowId * vocabSize;
    float const negInf = -std::numeric_limits<float>::infinity();
    for (int32_t i = tid; i < vocabSize; i += BLOCK_SIZE_)
    {
        outRow[i] = negInf;
    }
    __syncthreads();

    // Step 2: k-round block-reduction over the k * BLOCKS_PER_BEAM_ valid candidates.
    // (Only the first 'size' entries of the row's tmp buffer were written by Stage 1.)
    TopK_2<float> partial;
    __shared__ float sMaxLogit;
    for (int32_t ite = 0; ite < k; ite++)
    {
        partial.init();
        for (int32_t i = tid; i < size; i += BLOCK_SIZE_)
        {
            partial.insert(static_cast<float>(sVal[i]), i);
        }

        TopK_2<float> total = BlockReduce(tempStorage).Reduce(partial, reduce_topk_op_2<float>);

        if (tid == 0)
        {
            if (ite == 0)
            {
                sMaxLogit = total.u;
            }
            sId[ite] = total.p;
            sVal[total.p] = -MAX_T_VAL; // zero out so next iteration finds next-best
            sVal2[ite] = total.u;       // store raw logit value (not exponentiated)
        }
        __syncthreads();
    }

    // Step 3: Determine top-p cutoff (tid=0 only).
    // sVal2 contains logit values in descending order; we exponentiate to get unnormalized probs.
    if (tid == 0)
    {
        int32_t cutoff = k;
        if (topPs != nullptr)
        {
            float const topP = topPs[rowId];
            if (topP < 1.0f)
            {
                // Compute unnormalized probabilities and their sum.
                float sSum = 0.0f;
                for (int32_t ki = 0; ki < k; ki++)
                {
                    sVal2[ki] = __expf(sVal2[ki] - sMaxLogit); // reuse sVal2 to hold exp probs
                    sSum += sVal2[ki];
                }
                // Walk in descending-probability order; stop as soon as cumulative prob >= topP.
                float cumProb = 0.0f;
                for (int32_t ki = 0; ki < k; ki++)
                {
                    cumProb += sVal2[ki] / sSum;
                    if (cumProb >= topP)
                    {
                        cutoff = ki + 1; // always keep at least this token
                        break;
                    }
                }
            }
        }

        // Step 4: Scatter selected logit values back to output.
        // topKTmpIdBuf stores (rowOffset + vocabIdx); recover vocabIdx with % vocabSize.
        auto const rowStride = rowId * stride;
        for (int32_t ki = 0; ki < cutoff; ki++)
        {
            auto const candidateIdx = sId[ki];
            auto const globalIdx = topKTmpIdBuf[rowStride + candidateIdx];
            if (globalIdx >= 0)
            {
                auto const vocabIdx = globalIdx % vocabSize;
                // sVal2 was overwritten with exp probs when topP < 1; we need the original logit.
                // Re-read from the original tmp buffer — the stored value IS the logit (set in Stage 1).
                // However sVal[candidateIdx] was zeroed during Stage 2 reduction; but
                // topKTmpValBuf still holds the original value at that index (sVal points there).
                // We stored the logit as sVal2[ite] = total.u BEFORE any exp, so if topP was not
                // applied we can use sVal2[ki] directly. If topP was applied, sVal2[ki] now holds
                // the exp prob — we cannot recover the logit. To handle both cases cleanly we use
                // log(sVal2[ki]) + sMaxLogit when topPs was applied, otherwise sVal2[ki] directly.
                float logitVal;
                if (topPs != nullptr && topPs[rowId] < 1.0f)
                {
                    // sVal2[ki] = exp(logit - sMaxLogit), so logit = log(sVal2[ki]) + sMaxLogit
                    logitVal = __logf(sVal2[ki]) + sMaxLogit;
                }
                else
                {
                    logitVal = sVal2[ki]; // still the raw logit
                }
                outRow[vocabIdx] = logitVal;
            }
        }
    }
}

#define CASE_K_PROB(K_MAX, BLOCK_SIZE_1_, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_)                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        topKProbStage1<T, BLOCK_SIZE_1_, BLOCKS_PER_BEAM_>                                                             \
            <<<dim3(nRows* BLOCKS_PER_BEAM_, 1), BLOCK_SIZE_1_, 0, stream>>>(                                          \
                logits, tmpLogProbs, topKTmpIdBuf, topKTmpValBuf, maxTopK, topKs, vocabSize);                          \
        topKProbStage2ForLogits<T, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_>                                                    \
            <<<dim3(nRows, 1), BLOCK_SIZE_2_, K_MAX * (sizeof(int32_t) + sizeof(float)), stream>>>(                    \
                topKTmpIdBuf, topKTmpValBuf, outputLogits, maxTopK, topKs, topPs, vocabSize);                          \
    } while (0)

// Host launcher: allocates workspace tensors internally and dispatches the two-stage kernels.
// logits        [nRows, vocabSize] – temperature-scaled input (float or half)
// outputLogits  [nRows, vocabSize] – output: -inf everywhere except selected top-k-p positions
// topKs         [nRows]            – per-row k values (int32, on device)
// topPs         [nRows] or nullptr – per-row p values (float, on device)
// maxTopK                          – maximum k across all rows (CPU scalar, 1–1024)
template <typename T>
void invokeTopKTopPMaskingForProbs(T const* logits, float* outputLogits, int32_t const* topKs, float const* topPs,
    int32_t maxTopK, int32_t nRows, int32_t vocabSize, cudaStream_t stream)
{
    constexpr int32_t BLOCKS_PER_BEAM = 8;

    // Workspace buffers (allocated as CUDA device tensors via ATen).
    auto opts = at::TensorOptions().dtype(torch::kFloat32).device(at::kCUDA);
    auto tmpLogProbsTensor = torch::empty({nRows * vocabSize}, opts);
    auto topKTmpIdBufTensor
        = torch::empty({nRows * BLOCKS_PER_BEAM * maxTopK}, at::TensorOptions().dtype(torch::kInt32).device(at::kCUDA));
    // topKTmpValBuf uses the same dtype as T; we allocate as float and reinterpret for half if needed.
    auto topKTmpValBufTensor = torch::empty({nRows * BLOCKS_PER_BEAM * maxTopK}, opts);

    T* tmpLogProbs = reinterpret_cast<T*>(tmpLogProbsTensor.data_ptr<float>());
    int32_t* topKTmpIdBuf = topKTmpIdBufTensor.data_ptr<int32_t>();
    T* topKTmpValBuf = reinterpret_cast<T*>(topKTmpValBufTensor.data_ptr<float>());

    int32_t logMaxTopK = 0;
    int32_t recursor = maxTopK - 1;
    while (recursor >>= 1)
    {
        ++logMaxTopK;
    }

    switch (logMaxTopK)
    {
    case 0:
    case 1:
    case 2:
    case 3: // 0 < maxTopK <= 16
        CASE_K_PROB(16, 128, 128, 8);
        break;
    case 4: // 16 < maxTopK <= 32
        CASE_K_PROB(32, 256, 128, 8);
        break;
    case 5: // 32 < maxTopK <= 64
        CASE_K_PROB(64, 256, 256, 8);
        break;
    case 6:
    case 7:
    case 8:
    case 9: // 64 < maxTopK <= 1024
        CASE_K_PROB(1024, 256, 256, 8);
        break;
    default: TLLM_CHECK_WITH_INFO(false, "topKProbMasking supports 1 <= k <= 1024 but got k=%d", maxTopK);
    }
}

#undef CASE_K_PROB

namespace
{
constexpr double kGreedyTempThreshold = 1e-4;

bool isTopPEnabled(torch::optional<torch::Tensor> const& topP)
{
    return topP.has_value() && topP->defined() && topP->lt(1.0).any().item<bool>();
}

torch::Tensor computeSoftmaxForProbOp(torch::Tensor logits)
{
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
    TORCH_CHECK(logits.dim() == 2, "logits must be a 2D tensor");

    auto probs = logits.contiguous().to(torch::kFloat32);
    auto stream = at::cuda::getCurrentCUDAStream(probs.device().index());

    BiasSoftmaxParams<float> biasSoftmaxParams;
    biasSoftmaxParams.logits = probs.data_ptr<float>();
    biasSoftmaxParams.probs = probs.data_ptr<float>();
    biasSoftmaxParams.batchSize = static_cast<SizeType32>(probs.size(0));
    biasSoftmaxParams.maxBatchSize = static_cast<SizeType32>(probs.size(0));
    biasSoftmaxParams.maxBeamWidth = 1;
    biasSoftmaxParams.vocabSize = static_cast<SizeType32>(probs.size(1));
    biasSoftmaxParams.vocabSizePadded = static_cast<SizeType32>(probs.size(1));
    biasSoftmaxParams.skipSoftMax = false;
    biasSoftmaxParams.batchSlotsLogits = false;
    biasSoftmaxParams.checkParams();

    invokeAddBiasSoftMax(biasSoftmaxParams, stream);
    return probs;
}

struct DraftProbMaxFloatOp
{
    __device__ __forceinline__ float operator()(float a, float b) const
    {
        return a > b ? a : b;
    }
};

template <int32_t BLOCK_SIZE>
__global__ void computeDraftProbsSkipAllKernel(float const* draftLogits, int32_t const* d2t, float* draftProbs,
    int32_t nRows, int32_t draftVocabSize, int32_t targetVocabSize)
{
    int32_t const rowId = static_cast<int32_t>(blockIdx.x);
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    if (rowId >= nRows)
    {
        return;
    }

    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ float sMaxLogit;
    __shared__ float sExpSum;

    float const* rowLogits = draftLogits + static_cast<int64_t>(rowId) * draftVocabSize;
    float* rowProbs = draftProbs + static_cast<int64_t>(rowId) * targetVocabSize;

    for (int32_t v = tid; v < targetVocabSize; v += BLOCK_SIZE)
    {
        rowProbs[v] = 0.0f;
    }

    float localMax = -FLT_MAX;
    for (int32_t v = tid; v < draftVocabSize; v += BLOCK_SIZE)
    {
        localMax = fmaxf(localMax, rowLogits[v]);
    }

    float const blockMax = BlockReduce(tempStorage).Reduce(localMax, DraftProbMaxFloatOp{});
    if (tid == 0)
    {
        sMaxLogit = blockMax;
    }
    __syncthreads();

    float localSum = 0.0f;
    for (int32_t v = tid; v < draftVocabSize; v += BLOCK_SIZE)
    {
        localSum += __expf(rowLogits[v] - sMaxLogit);
    }

    float const blockSum = BlockReduce(tempStorage).Sum(localSum);
    if (tid == 0)
    {
        constexpr float kFloatSoftmaxEpsilon = 1e-6f;
        sExpSum = blockSum + kFloatSoftmaxEpsilon;
    }
    __syncthreads();

    for (int32_t v = tid; v < draftVocabSize; v += BLOCK_SIZE)
    {
        int64_t const targetIdx
            = d2t != nullptr ? static_cast<int64_t>(v) + static_cast<int64_t>(d2t[v]) : static_cast<int64_t>(v);
        if (targetIdx >= 0 && targetIdx < targetVocabSize)
        {
            rowProbs[targetIdx] = __expf(rowLogits[v] - sMaxLogit) / sExpSum;
        }
    }
}

torch::Tensor computeDraftProbsSkipAllForDynamicTreeRejection(torch::Tensor const& draftLogits, int64_t batchSize,
    SizeType32 const numDraftProbRows, SizeType32 const targetVocabSize, torch::optional<torch::Tensor> const& d2t)
{
    auto const draftVocabSize = draftLogits.size(1);
    bool const hasD2T = d2t.has_value() && d2t->defined();

    auto draftLogitsFloat = draftLogits.contiguous().to(torch::kFloat32);
    if (!hasD2T && draftVocabSize == targetVocabSize)
    {
        return computeSoftmaxForProbOp(draftLogitsFloat).reshape({batchSize, numDraftProbRows, targetVocabSize});
    }

    auto fullDraftProbs = torch::empty({draftLogitsFloat.size(0), targetVocabSize},
        torch::TensorOptions().dtype(torch::kFloat32).device(draftLogitsFloat.device()));
    torch::Tensor d2tInt;
    int32_t const* d2tPtr = nullptr;
    if (hasD2T)
    {
        d2tInt = d2t->contiguous().to(torch::kInt32);
        d2tPtr = d2tInt.data_ptr<int32_t>();
    }

    constexpr int32_t kBlockSize = 1024;
    dim3 grid(draftLogitsFloat.size(0));
    dim3 block(kBlockSize);
    auto stream = at::cuda::getCurrentCUDAStream(draftLogitsFloat.device().index());
    computeDraftProbsSkipAllKernel<kBlockSize><<<grid, block, 0, stream>>>(draftLogitsFloat.data_ptr<float>(), d2tPtr,
        fullDraftProbs.data_ptr<float>(), static_cast<int32_t>(draftLogitsFloat.size(0)),
        static_cast<int32_t>(draftVocabSize), static_cast<int32_t>(targetVocabSize));
    sync_check_cuda_error(stream);

    return fullDraftProbs.reshape({batchSize, numDraftProbRows, targetVocabSize});
}

// Fast path for top-K (and optional top-P) filtering using torch::topk instead of a
// full vocab-size sort.  kMax must be provided as a CPU integer (the caller computes it
// via topK.max().item() on the Python side).  When kMax == 0 or kMax >= vocabSize the
// function falls back to the original sort-based path.
//
// Key advantages over the full-sort path:
//   1. torch::topk with small kMax is O(V * log kMax) vs O(V * log V) for full sort.
//   2. The topk index tensor is [nRows, kMax] instead of [nRows, V] — much smaller.
//   3. No scatter-back of sorted indices needed; masking is done directly on logits.
//   4. For combined top-K + top-P, softmax/cumsum are computed on kMax values (not V).
torch::Tensor applyTopKTopPForProbOp(torch::Tensor logits, torch::optional<torch::Tensor> const& topK,
    torch::optional<torch::Tensor> const& topP, int32_t kMax)
{
    int64_t const vocabSize = logits.size(1);
    // Host-only checks: the caller is expected to pass nullopt when filtering is fully
    // disabled (see SpecMetadata.skip_top_k / skip_top_p). Probing the tensor contents
    // via `.item<bool>()` here would force a host-device sync and break CUDA graph
    // capture; the per-row `effectiveTopK` formula below already handles disabled rows.
    bool const hasTopK = topK.has_value() && topK->defined();
    bool const hasTopP = topP.has_value() && topP->defined();

    if (!hasTopK && !hasTopP)
    {
        return logits;
    }

    torch::Tensor effectiveTopK;
    if (hasTopK)
    {
        auto topKLong = topK->to(torch::kLong);
        effectiveTopK
            = torch::where(topKLong > 0, topKLong, torch::full_like(topKLong, vocabSize)).clamp_max(vocabSize);
    }

    // Fast path uses `topk(kMax)` which is unsafe when any row has effective top-k > kMax
    // (i.e. disabled rows expand to the full vocab). Detecting this requires a tensor
    // reduction + `.item<bool>()`, which is incompatible with CUDA graph capture. Only
    // probe when the caller explicitly opted into the fast path via kMax > 0 (today only
    // the dynamic-tree caller, which is not graph-captured).
    bool hasDisabledTopKRows = false;
    if (hasTopK && kMax > 0 && kMax < vocabSize)
    {
        auto topKLong = topK->to(torch::kLong);
        hasDisabledTopKRows = topKLong.le(0).any().item<bool>();
    }

    if (hasTopK && !hasDisabledTopKRows && kMax > 0 && kMax < vocabSize)
    {
        // Fast topk path ─────────────────────────────────────────────────────────────
        // topKValues/topKIdx: [nRows, kMax], values in descending order
        auto [topKValues, topKIdx] = logits.topk(kMax, /*dim=*/-1, /*largest=*/true, /*sorted=*/true);

        // validTopK[i, j]: True when position j falls within top-K[i] for row i
        auto kArange = torch::arange(kMax, torch::TensorOptions().dtype(torch::kInt64).device(logits.device()))
                           .unsqueeze(0);                          // [1, kMax]
        auto kVals = effectiveTopK.to(torch::kInt64).unsqueeze(1); // [nRows, 1]
        auto validTopK = kArange < kVals;                          // [nRows, kMax]

        // Start with everything masked; scatter will unmark the kept positions.
        auto mask = torch::ones(
            {logits.size(0), vocabSize}, torch::TensorOptions().dtype(torch::kBool).device(logits.device()));

        if (hasTopP)
        {
            // Compute top-P on the kMax descending-sorted values only (much cheaper).
            // Positions beyond K[i] are treated as -inf so their probability ≈ 0.
            auto validTopKValues = topKValues.masked_fill(~validTopK, -std::numeric_limits<float>::infinity());
            auto sortedProbs = validTopKValues.softmax(/*dim=*/-1); // [nRows, kMax]
            auto cumsum = sortedProbs.cumsum(/*dim=*/-1);           // [nRows, kMax]
            // Mask positions where the cumulative probability *before* this token
            // already reaches topP — i.e. we have enough probability mass already.
            auto topPMask = (cumsum - sortedProbs) >= topP->unsqueeze(1); // [nRows, kMax]
            topPMask.select(/*dim=*/1, /*index=*/0).fill_(false);         // always keep the top-1 token
            // combinedMask: True  → mask this vocab position
            //               False → keep this vocab position
            auto combinedMask = topPMask | (~validTopK); // [nRows, kMax]
            mask.scatter_(/*dim=*/1, /*index=*/topKIdx, /*src=*/combinedMask);
        }
        else
        {
            // Top-K only: unmark the first K[i] positions (those within validTopK).
            // ~validTopK is True for positions j >= K[i] → they should stay masked.
            mask.scatter_(/*dim=*/1, /*index=*/topKIdx, /*src=*/(~validTopK));
        }

        return logits.masked_fill(mask, -std::numeric_limits<float>::infinity());
    }

    // Fallback: full-sort path (used for top-P only, or when kMax == 0) ────────────
    auto sortResult = logits.sort(/*dim=*/-1, /*descending=*/false);
    auto logitsSort = std::get<0>(sortResult);
    auto logitsIdx = std::get<1>(sortResult);

    if (hasTopK)
    {
        auto topKMask = logitsSort.size(1) - effectiveTopK;
        topKMask = topKMask.clamp_min(0);
        auto topKThreshold = logitsSort.gather(1, topKMask.unsqueeze(1));
        auto mask = logitsSort < topKThreshold;
        logitsSort.masked_fill_(mask, -std::numeric_limits<float>::infinity());
    }

    if (hasTopP)
    {
        auto probsSort = logitsSort.softmax(/*dim=*/-1);
        auto probsSum = probsSort.cumsum(/*dim=*/-1, /*dtype=*/probsSort.scalar_type());
        auto topPMask = probsSum <= (1.0 - topP->unsqueeze(1));
        topPMask.select(/*dim=*/1, /*index=*/logitsSort.size(1) - 1).fill_(false);
        logitsSort.masked_fill_(topPMask, -std::numeric_limits<float>::infinity());
    }

    return logitsSort.scatter(/*dim=*/-1, /*index=*/logitsIdx, /*src=*/logitsSort);
}

} // namespace

torch::Tensor computeProbsFromLogits(torch::Tensor const& logits, torch::Tensor const& temperatures,
    torch::optional<torch::Tensor> const& topK, torch::optional<torch::Tensor> const& topP, bool skipTemperature,
    int32_t kMax)
{
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
    TORCH_CHECK(temperatures.is_cuda(), "temperatures must be a CUDA tensor");
    TORCH_CHECK(logits.dim() == 2, "logits must be a 2D tensor");
    TORCH_CHECK(temperatures.dim() == 1, "temperatures must be a 1D tensor");
    TORCH_CHECK(logits.size(0) == temperatures.size(0), "logits and temperatures size mismatch");
    if (topK.has_value() && topK->defined())
    {
        TORCH_CHECK(topK->is_cuda(), "top_k must be a CUDA tensor");
        TORCH_CHECK(topK->dim() == 1, "top_k must be a 1D tensor");
        TORCH_CHECK(topK->size(0) == logits.size(0), "top_k and logits size mismatch");
    }
    if (topP.has_value() && topP->defined())
    {
        TORCH_CHECK(topP->is_cuda(), "top_p must be a CUDA tensor");
        TORCH_CHECK(topP->dim() == 1, "top_p must be a 1D tensor");
        TORCH_CHECK(topP->size(0) == logits.size(0), "top_p and logits size mismatch");
    }

    auto const isGreedy = temperatures <= kGreedyTempThreshold;
    auto const safeTemperatures = torch::where(isGreedy, torch::ones_like(temperatures), temperatures);
    auto scaledLogits
        = (skipTemperature ? logits : logits.div(safeTemperatures.unsqueeze(1))).contiguous().to(torch::kFloat32);

    int64_t const vocabSize = scaledLogits.size(1);
    int64_t const nRows = scaledLogits.size(0);
    // Host-only presence checks; see comment in applyTopKTopPForProbOp() for why we
    // avoid probing tensor contents (would sync and break CUDA graph capture).
    bool const hasTopKPresence = topK.has_value() && topK->defined();
    bool const hasTopPPresence = topP.has_value() && topP->defined();

    // The kernel path produces -inf for rows whose top_k value is 0, so it is only
    // safe when every row has an active top_k filter. Determining that requires a
    // host-device sync, so only probe when the caller has opted into the kernel
    // path (kMax > 0). The kMax > 0 callers (dynamic-tree) are not graph-captured.
    bool useKernelPath = false;
    if (hasTopKPresence && kMax > 0 && kMax < vocabSize)
    {
        useKernelPath = torch::logical_and(topK->gt(0), topK->lt(vocabSize)).any().item<bool>();
    }

    torch::Tensor maskedLogits;
    if (useKernelPath)
    {
        // Two-stage CUDA top-k/top-p masking (mirrors invokeBatchTopKSampling).
        maskedLogits = torch::empty_like(scaledLogits);
        auto topKForKernel = topK->to(torch::kInt32).contiguous();
        auto topPForKernel = hasTopPPresence ? topP->to(torch::kFloat32).contiguous() : torch::Tensor();
        auto stream = at::cuda::getCurrentCUDAStream(scaledLogits.device().index());
        invokeTopKTopPMaskingForProbs<float>(scaledLogits.data_ptr<float>(), maskedLogits.data_ptr<float>(),
            topKForKernel.data_ptr<int32_t>(), hasTopPPresence ? topPForKernel.data_ptr<float>() : nullptr, kMax,
            static_cast<int32_t>(nRows), static_cast<int32_t>(vocabSize), stream);
    }
    else
    {
        // Fallback: PyTorch-based sort path (top-P only or kMax == 0).
        maskedLogits = applyTopKTopPForProbOp(scaledLogits, topK, topP, kMax);
    }

    auto probs = computeSoftmaxForProbOp(maskedLogits);

    auto argmaxIds = maskedLogits.argmax(/*dim=*/-1, /*keepdim=*/true);
    auto oneHot = torch::zeros_like(probs).scatter_(1, argmaxIds, 1.0);
    return torch::where(isGreedy.unsqueeze(1), oneHot, probs);
}

torch::Tensor computeDraftProbsForDynamicTreeRejection(torch::Tensor const& draftLogits,
    torch::Tensor const& temperatures, SizeType32 const numDraftProbRows, torch::optional<torch::Tensor> const& topK,
    torch::optional<torch::Tensor> const& topP, SizeType32 const targetVocabSize, bool skipTemperature,
    torch::optional<torch::Tensor> const& d2t, SizeType32 const kMax, bool skipAllSamplingParams)
{
    TORCH_CHECK(draftLogits.is_cuda(), "draftLogits must be a CUDA tensor");
    TORCH_CHECK(temperatures.is_cuda(), "temperatures must be a CUDA tensor");
    TORCH_CHECK(draftLogits.dim() == 2, "draftLogits must be a 2D tensor");
    TORCH_CHECK(temperatures.dim() == 1, "temperatures must be a 1D tensor");
    TORCH_CHECK(numDraftProbRows > 0, "numDraftProbRows must be positive");

    auto const batchSize = temperatures.size(0);
    auto const draftVocabSize = draftLogits.size(1);

    TORCH_CHECK(batchSize > 0, "batchSize must be positive");
    TORCH_CHECK(
        draftLogits.size(0) == batchSize * numDraftProbRows, "draftLogits row count does not match numDraftProbRows");
    TORCH_CHECK(targetVocabSize >= draftVocabSize, "targetVocabSize must be >= draft vocab size");

    if (topK.has_value() && topK->defined())
    {
        TORCH_CHECK(topK->is_cuda(), "top_k must be a CUDA tensor");
        TORCH_CHECK(topK->dim() == 1, "top_k must be a 1D tensor");
        TORCH_CHECK(topK->size(0) == batchSize, "top_k size mismatch");
    }
    if (topP.has_value() && topP->defined())
    {
        TORCH_CHECK(topP->is_cuda(), "top_p must be a CUDA tensor");
        TORCH_CHECK(topP->dim() == 1, "top_p must be a 1D tensor");
        TORCH_CHECK(topP->size(0) == batchSize, "top_p size mismatch");
    }
    if (d2t.has_value() && d2t->defined())
    {
        TORCH_CHECK(d2t->is_cuda(), "d2t must be a CUDA tensor");
        TORCH_CHECK(d2t->dim() == 1, "d2t must be a 1D tensor");
        TORCH_CHECK(d2t->size(0) >= draftVocabSize, "d2t size mismatch");
    }

    if (skipAllSamplingParams)
    {
        return computeDraftProbsSkipAllForDynamicTreeRejection(
            draftLogits, batchSize, numDraftProbRows, targetVocabSize, d2t);
    }

    auto draftTemps = temperatures.repeat_interleave(numDraftProbRows);
    auto draftTopK = topK.has_value() && topK->defined()
        ? torch::optional<torch::Tensor>(topK->repeat_interleave(numDraftProbRows))
        : torch::optional<torch::Tensor>();
    auto draftTopP = isTopPEnabled(topP) ? torch::optional<torch::Tensor>(topP->repeat_interleave(numDraftProbRows))
                                         : torch::optional<torch::Tensor>();

    auto draftProbs = computeProbsFromLogits(draftLogits, draftTemps, draftTopK, draftTopP, skipTemperature, kMax)
                          .reshape({batchSize, numDraftProbRows, draftVocabSize});

    if (draftVocabSize == targetVocabSize)
    {
        return draftProbs;
    }

    auto fullDraftProbs = torch::zeros({batchSize, numDraftProbRows, targetVocabSize},
        torch::TensorOptions().dtype(torch::kFloat32).device(draftProbs.device()));
    if (d2t.has_value() && d2t->defined())
    {
        auto srcIdx
            = torch::arange(draftVocabSize, torch::TensorOptions().dtype(torch::kInt64).device(draftProbs.device()));
        auto targetIdx = srcIdx + d2t->slice(0, 0, draftVocabSize).to(torch::kInt64);
        auto expandedTargetIdx
            = targetIdx.view({1, 1, draftVocabSize}).expand({batchSize, numDraftProbRows, draftVocabSize});
        fullDraftProbs.scatter_(2, expandedTargetIdx, draftProbs);
    }
    else
    {
        fullDraftProbs.slice(/*dim=*/2, /*start=*/0, /*end=*/draftVocabSize).copy_(draftProbs);
    }

    return fullDraftProbs;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> computeTargetProbsForDynamicTreeRejection(
    torch::Tensor const& targetLogits, torch::Tensor const& temperatures, SizeType32 const numDraftTokens,
    torch::optional<torch::Tensor> const& topK, torch::optional<torch::Tensor> const& topP, bool skipTemperature,
    SizeType32 const kMax, bool skipAllSamplingParams)
{
    TORCH_CHECK(targetLogits.is_cuda(), "targetLogits must be a CUDA tensor");
    TORCH_CHECK(temperatures.is_cuda(), "temperatures must be a CUDA tensor");
    TORCH_CHECK(targetLogits.dim() == 2, "targetLogits must be a 2D tensor");
    TORCH_CHECK(temperatures.dim() == 1, "temperatures must be a 1D tensor");
    TORCH_CHECK(numDraftTokens > 1, "numDraftTokens must be greater than 1");

    auto const batchSize = temperatures.size(0);
    auto const targetVocabSize = targetLogits.size(1);
    auto const nRows = batchSize * numDraftTokens;

    TORCH_CHECK(batchSize > 0, "batchSize must be positive");
    TORCH_CHECK(
        targetLogits.size(0) == batchSize * numDraftTokens, "targetLogits row count does not match numDraftTokens");

    if (topK.has_value() && topK->defined())
    {
        TORCH_CHECK(topK->is_cuda(), "top_k must be a CUDA tensor");
        TORCH_CHECK(topK->dim() == 1, "top_k must be a 1D tensor");
        TORCH_CHECK(topK->size(0) == batchSize, "top_k size mismatch");
    }
    if (topP.has_value() && topP->defined())
    {
        TORCH_CHECK(topP->is_cuda(), "top_p must be a CUDA tensor");
        TORCH_CHECK(topP->dim() == 1, "top_p must be a 1D tensor");
        TORCH_CHECK(topP->size(0) == batchSize, "top_p size mismatch");
    }

    if (skipAllSamplingParams)
    {
        auto targetSupportIndices
            = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(targetLogits.device()));
        auto targetSupportLengths
            = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(targetLogits.device()));
        auto targetProbs = computeSoftmaxForProbOp(targetLogits);
        return std::make_tuple(targetProbs.reshape({batchSize, numDraftTokens, targetVocabSize}), targetSupportIndices,
            targetSupportLengths);
    }

    auto targetTemps = temperatures.repeat_interleave(numDraftTokens);
    auto targetTopK = topK.has_value() && topK->defined()
        ? torch::optional<torch::Tensor>(topK->repeat_interleave(numDraftTokens))
        : torch::optional<torch::Tensor>();
    auto targetTopP = isTopPEnabled(topP) ? torch::optional<torch::Tensor>(topP->repeat_interleave(numDraftTokens))
                                          : torch::optional<torch::Tensor>();

    bool const hasTopK = targetTopK.has_value() && targetTopK->defined();
    bool const hasTopP = isTopPEnabled(targetTopP);
    bool const hasFiltering = hasTopK || hasTopP;
    torch::Tensor effectiveTargetTopK;
    bool hasDisabledTopKRows = false;

    auto const isGreedy = targetTemps <= kGreedyTempThreshold;
    auto const safeTargetTemps = torch::where(isGreedy, torch::ones_like(targetTemps), targetTemps);
    auto scaledTargetLogits = (skipTemperature ? targetLogits : targetLogits.div(safeTargetTemps.unsqueeze(1)))
                                  .contiguous()
                                  .to(torch::kFloat32);

    if (hasTopK)
    {
        auto targetTopKLong = targetTopK->to(torch::kLong);
        effectiveTargetTopK
            = torch::where(targetTopKLong > 0, targetTopKLong, torch::full_like(targetTopKLong, targetVocabSize))
                  .clamp_max(targetVocabSize);
        hasDisabledTopKRows = targetTopKLong.le(0).any().item<bool>();
    }

    torch::Tensor maskedTargetLogits;
    torch::Tensor targetSupportIndices;
    torch::Tensor targetSupportLengths;

    if (!hasFiltering)
    {
        // No filtering: use full-vocab probs; sparse support is not applicable.
        maskedTargetLogits = scaledTargetLogits;
        targetSupportIndices
            = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(targetLogits.device()));
        targetSupportLengths
            = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(targetLogits.device()));
    }
    else if (hasTopK && !hasDisabledTopKRows && kMax > 0 && static_cast<int64_t>(kMax) < targetVocabSize)
    {
        // Fast two-stage CUDA path for masked logits.
        maskedTargetLogits = torch::empty_like(scaledTargetLogits);
        auto topKForKernel = effectiveTargetTopK.to(torch::kInt32).contiguous();
        auto topPForKernel = hasTopP ? targetTopP->to(torch::kFloat32).contiguous() : torch::Tensor();
        auto stream = at::cuda::getCurrentCUDAStream(scaledTargetLogits.device().index());
        invokeTopKTopPMaskingForProbs<float>(scaledTargetLogits.data_ptr<float>(), maskedTargetLogits.data_ptr<float>(),
            topKForKernel.data_ptr<int32_t>(), hasTopP ? topPForKernel.data_ptr<float>() : nullptr, kMax,
            static_cast<int32_t>(nRows), static_cast<int32_t>(targetVocabSize), stream);

        // Extract support indices: the finite positions after masking (at most kMax per row).
        auto [topKVals, topKIdx] = maskedTargetLogits.topk(kMax, /*dim=*/-1, /*largest=*/true, /*sorted=*/true);
        auto validMask = topKVals.isfinite();                                                 // [nRows, kMax]
        auto supportLengthsLong = validMask.sum(/*dim=*/-1, /*keepdim=*/false, torch::kLong); // [nRows]
        auto supportIndicesRaw
            = torch::where(validMask, topKIdx.to(torch::kInt32), torch::full_like(topKIdx.to(torch::kInt32), -1));

        targetSupportIndices = supportIndicesRaw.reshape({batchSize, numDraftTokens, kMax});
        targetSupportLengths = supportLengthsLong.to(torch::kInt32).reshape({batchSize, numDraftTokens});
    }
    else
    {
        // Sort-based fallback: top-P only, or kMax == 0 / kMax >= vocabSize.
        auto sortResult = scaledTargetLogits.sort(/*dim=*/-1, /*descending=*/false);
        auto logitsSort = std::get<0>(sortResult);
        auto logitsIdx = std::get<1>(sortResult);

        if (hasTopK)
        {
            auto topKMask = logitsSort.size(1) - effectiveTargetTopK;
            topKMask = topKMask.clamp_min(0);
            auto topKThreshold = logitsSort.gather(1, topKMask.unsqueeze(1));
            auto mask = logitsSort < topKThreshold;
            logitsSort.masked_fill_(mask, -std::numeric_limits<float>::infinity());
        }

        if (hasTopP)
        {
            auto probsSort = logitsSort.softmax(/*dim=*/-1);
            auto probsSum = probsSort.cumsum(/*dim=*/-1, /*dtype=*/probsSort.scalar_type());
            auto topPMask = probsSum <= (1.0 - targetTopP->unsqueeze(1));
            topPMask.select(/*dim=*/1, /*index=*/logitsSort.size(1) - 1).fill_(false);
            logitsSort.masked_fill_(topPMask, -std::numeric_limits<float>::infinity());
        }

        maskedTargetLogits = logitsSort.scatter(/*dim=*/-1, /*index=*/logitsIdx, /*src=*/logitsSort);

        // Compact support indices: finite values are at the END of the ascending-sorted logitsSort.
        auto supportLengthsLong
            = logitsSort.isfinite().sum(/*dim=*/-1, /*keepdim=*/false, /*dtype=*/torch::kLong); // [nRows]
        auto supportLengths1D = supportLengthsLong.to(torch::kInt32);

        int64_t maxSupportSize = targetVocabSize;
        if (hasTopK && effectiveTargetTopK.defined())
        {
            maxSupportSize = std::min<int64_t>(targetVocabSize, effectiveTargetTopK.max().item<int64_t>());
        }

        auto compactPositions
            = torch::arange(maxSupportSize, torch::TensorOptions().dtype(torch::kLong).device(targetLogits.device()))
                  .unsqueeze(0)
                  .expand({nRows, maxSupportSize});
        auto supportStart = targetVocabSize - supportLengthsLong.unsqueeze(1);
        auto gatherPositions = (supportStart + compactPositions).clamp_max(targetVocabSize - 1);
        auto gatheredSupportIndices = logitsIdx.gather(1, gatherPositions);
        auto validMask = compactPositions < supportLengthsLong.unsqueeze(1);
        auto invalidFill = torch::full_like(gatheredSupportIndices, -1L);
        targetSupportIndices = torch::where(validMask, gatheredSupportIndices, invalidFill)
                                   .to(torch::kInt32)
                                   .reshape({batchSize, numDraftTokens, maxSupportSize});
        targetSupportLengths = supportLengths1D.reshape({batchSize, numDraftTokens});
    }

    auto targetProbs = computeSoftmaxForProbOp(maskedTargetLogits);

    auto argmaxIds = maskedTargetLogits.argmax(/*dim=*/-1, /*keepdim=*/true);
    auto oneHot = torch::zeros_like(targetProbs).scatter_(1, argmaxIds, 1.0);
    targetProbs = torch::where(isGreedy.unsqueeze(1), oneHot, targetProbs);

    return std::make_tuple(
        targetProbs.reshape({batchSize, numDraftTokens, targetVocabSize}), targetSupportIndices, targetSupportLengths);
}

//! \param parentList           [in]  layer-wise parent indices [bs, topK*(depth-1)+1]
//! \param selectedIndex        [in]  resampled history buffer indices [bs, draftTokenNum-1]
//! \param treeMask             [out] attention mask (which nodes each node can see)
//! \param positions            [out] position id per node [bs, draftTokenNum]
//! \param retrieveIndex        [out] tree node -> local index mapping [bs, draftTokenNum]
//! \param retrieveNextToken    [out] first-child pointer [bs, draftTokenNum], -1=none
//! \param retrieveNextSibling  [out] next-sibling pointer [bs, draftTokenNum], -1=none
//! \param topK                 top-K value per layer
//! \param depth                max tree depth (number of draft layers)
//! \param draftTokenNum        total tree nodes per batch (including root)
__global__ void buildDynamicTreeKernel(int64_t const* parentList, int64_t const* selectedIndex, int32_t* treeMask,
    int32_t* positions, int32_t* retrieveIndex, int32_t* retrieveNextToken, int32_t* retrieveNextSibling,
    SizeType32 topK, SizeType32 depth, SizeType32 draftTokenNum)
{
    int32_t bid = blockIdx.x;
    int32_t tid = threadIdx.x;

    if (tid >= draftTokenNum)
    {
        return;
    }

    // treeMask layout: [batchSize, draftTokenNum, draftTokenNum] (QLEN_ONLY mode)
    int32_t tokenTreeIdx = draftTokenNum * draftTokenNum * bid + draftTokenNum * tid + 1;

    treeMask[tokenTreeIdx - 1] = 1; // self-attention diagonal
    for (int32_t i = 0; i < draftTokenNum - 1; i++)
    {
        treeMask[tokenTreeIdx + i] = 0;
    }

    int32_t position = 0;

    if (tid == 0)
    {
        positions[bid * draftTokenNum] = 0;

        // Reverse iteration: inserting at list head produces forward sibling order
        for (int32_t i = draftTokenNum - 1; i > 0; --i)
        {
            retrieveIndex[bid * draftTokenNum + i] = i;

            int64_t parentTbIdx = selectedIndex[bid * (draftTokenNum - 1) + i - 1] / topK;
            int32_t parentPosition = 0;

            if (parentTbIdx > 0)
            {
                int64_t parentTokenIdx = parentList[bid * (topK * (depth - 1) + 1) + parentTbIdx];
                for (; parentPosition < draftTokenNum; ++parentPosition)
                {
                    if (selectedIndex[bid * (draftTokenNum - 1) + parentPosition] == parentTokenIdx)
                    {
                        ++parentPosition; // +1 because position 0 is root
                        break;
                    }
                }
            }

            if (parentPosition == draftTokenNum)
            {
                printf(
                    "WARNING: Invalid dynamic tree! Detected a token with no parent token selected. "
                    "Please check if the logprob has nan. The token will be ignored.\n");
                continue;
            }

            if (retrieveNextToken[bid * draftTokenNum + parentPosition] == -1)
            {
                retrieveNextToken[bid * draftTokenNum + parentPosition] = i;
            }
            else
            {
                int32_t originNextToken = retrieveNextToken[bid * draftTokenNum + parentPosition];
                retrieveNextToken[bid * draftTokenNum + parentPosition] = i;
                retrieveNextSibling[bid * draftTokenNum + i] = originNextToken;
            }
        }
        retrieveIndex[bid * draftTokenNum] = 0;
    }
    else
    {
        // Walk up to root, setting treeMask ancestor bits and counting depth
        int32_t curPosition = tid - 1;
        while (position < depth + 1)
        {
            position += 1;
            treeMask[tokenTreeIdx + curPosition] = 1;

            int64_t parentTbIdx = selectedIndex[bid * (draftTokenNum - 1) + curPosition] / topK;
            if (parentTbIdx == 0)
            {
                break;
            }

            int64_t tokenIdx = parentList[bid * (topK * (depth - 1) + 1) + parentTbIdx];
            for (curPosition = 0; curPosition < draftTokenNum; ++curPosition)
            {
                if (selectedIndex[bid * (draftTokenNum - 1) + curPosition] == tokenIdx)
                {
                    break;
                }
            }
            if (curPosition == draftTokenNum)
            {
                break;
            }
        }
        positions[bid * draftTokenNum + tid] = position;
    }
}

//! Bit-packed variant of buildDynamicTreeKernel.
//! \param numInt32PerRow  int32 count per treeMask row (buffer stride; >= ceil(draftTokenNum/32) if padded)
__global__ void buildDynamicTreeKernelPacked(int64_t const* parentList, int64_t const* selectedIndex, int32_t* treeMask,
    int32_t* positions, int32_t* retrieveIndex, int32_t* retrieveNextToken, int32_t* retrieveNextSibling,
    SizeType32 topK, SizeType32 depth, SizeType32 draftTokenNum, SizeType32 numInt32PerRow)
{
    int32_t bid = blockIdx.x;
    int32_t tid = threadIdx.x;

    if (tid >= draftTokenNum)
    {
        return;
    }

    int32_t rowBaseIdx = (bid * draftTokenNum + tid) * numInt32PerRow;

    treeMask[rowBaseIdx] = 1; // bit 0 = root, always visible

    int32_t position = 0;

    if (tid == 0)
    {
        positions[bid * draftTokenNum] = 0;

        for (int32_t i = draftTokenNum - 1; i > 0; --i)
        {
            retrieveIndex[bid * draftTokenNum + i] = i;

            int64_t parentTbIdx = selectedIndex[bid * (draftTokenNum - 1) + i - 1] / topK;
            int32_t parentPosition = 0;

            if (parentTbIdx > 0)
            {
                int64_t parentTokenIdx = parentList[bid * (topK * (depth - 1) + 1) + parentTbIdx];
                for (; parentPosition < draftTokenNum; ++parentPosition)
                {
                    if (selectedIndex[bid * (draftTokenNum - 1) + parentPosition] == parentTokenIdx)
                    {
                        ++parentPosition;
                        break;
                    }
                }
            }

            if (parentPosition == draftTokenNum)
            {
                printf("WARNING: Invalid dynamic tree! Detected a token with no parent token selected.\n");
                continue;
            }

            if (retrieveNextToken[bid * draftTokenNum + parentPosition] == -1)
            {
                retrieveNextToken[bid * draftTokenNum + parentPosition] = i;
            }
            else
            {
                int32_t originNextToken = retrieveNextToken[bid * draftTokenNum + parentPosition];
                retrieveNextToken[bid * draftTokenNum + parentPosition] = i;
                retrieveNextSibling[bid * draftTokenNum + i] = originNextToken;
            }
        }
        retrieveIndex[bid * draftTokenNum] = 0;
    }
    else
    {
        int32_t curPosition = tid - 1;
        while (position < depth + 1)
        {
            position += 1;

            int32_t bitPosition = curPosition + 1; // +1 because bit 0 is root
            int32_t int32Idx = bitPosition / 32;
            int32_t bitIdx = bitPosition % 32;

            if (int32Idx < numInt32PerRow)
            {
                atomicOr(&treeMask[rowBaseIdx + int32Idx], 1 << bitIdx);
            }

            int64_t parentTbIdx = selectedIndex[bid * (draftTokenNum - 1) + curPosition] / topK;
            if (parentTbIdx == 0)
            {
                break;
            }

            int64_t tokenIdx = parentList[bid * (topK * (depth - 1) + 1) + parentTbIdx];
            for (curPosition = 0; curPosition < draftTokenNum; ++curPosition)
            {
                if (selectedIndex[bid * (draftTokenNum - 1) + curPosition] == tokenIdx)
                {
                    break;
                }
            }
            if (curPosition == draftTokenNum)
            {
                break;
            }
        }
        positions[bid * draftTokenNum + tid] = position;
    }
}

void invokeBuildDynamicTree(int64_t const* parentList, int64_t const* selectedIndex, void* treeMask, int32_t* positions,
    int32_t* retrieveIndex, int32_t* retrieveNextToken, int32_t* retrieveNextSibling, SizeType32 batchSize,
    SizeType32 topK, SizeType32 depth, SizeType32 numDraftTokens, TreeMaskMode treeMaskMode, cudaStream_t stream,
    SizeType32 numInt32PerRow)
{
    dim3 grid(batchSize);
    dim3 block(numDraftTokens);

    if (treeMaskMode == TreeMaskMode::QLEN_ONLY_BITPACKING)
    {
        TLLM_CHECK_WITH_INFO(
            numInt32PerRow > 0, "numInt32PerRow must be the packed treeMask row stride in int32s (from buffer shape).");
        buildDynamicTreeKernelPacked<<<grid, block, 0, stream>>>(parentList, selectedIndex,
            static_cast<int32_t*>(treeMask), positions, retrieveIndex, retrieveNextToken, retrieveNextSibling, topK,
            depth, numDraftTokens, numInt32PerRow);
    }
    else
    {
        buildDynamicTreeKernel<<<grid, block, 0, stream>>>(parentList, selectedIndex, static_cast<int32_t*>(treeMask),
            positions, retrieveIndex, retrieveNextToken, retrieveNextSibling, topK, depth, numDraftTokens);
    }

    sync_check_cuda_error(stream);
}

__global__ void buildDraftProbIndicesKernel(
    int64_t const* topkScoreIndices, int32_t* draftProbIndices, SizeType32 topK, SizeType32 numDraftTokens)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const tokenIdx = threadIdx.x;

    if (tokenIdx > numDraftTokens)
    {
        return;
    }

    int32_t* draftProbIndicesRow = draftProbIndices + batchIdx * (numDraftTokens + 1);

    if (tokenIdx == 0)
    {
        draftProbIndicesRow[0] = 0;
        return;
    }

    int64_t const histIdx = topkScoreIndices[batchIdx * numDraftTokens + (tokenIdx - 1)];
    int32_t draftProbRow = 0;

    if (histIdx >= topK)
    {
        int64_t const relative = histIdx - topK;
        int64_t const depthBucket = relative / (topK * topK);
        int64_t const parentK = (relative % (topK * topK)) / topK;
        draftProbRow = static_cast<int32_t>(1 + depthBucket * topK + parentK);
    }

    draftProbIndicesRow[tokenIdx] = draftProbRow;
}

void invokeBuildDraftProbIndices(int64_t const* topkScoreIndices, int32_t* draftProbIndices, SizeType32 batchSize,
    SizeType32 topK, SizeType32 numDraftTokens, cudaStream_t stream)
{
    dim3 const grid(batchSize);
    dim3 const block(numDraftTokens + 1);

    buildDraftProbIndicesKernel<<<grid, block, 0, stream>>>(topkScoreIndices, draftProbIndices, topK, numDraftTokens);
    sync_check_cuda_error(stream);
}

//! \param predicts             [out] accepted token ids + bonus token [bs * numDraftTokens]
//! \param acceptIndex          [out] accepted path as local tree positions [bs, numSpeculativeTokens]
//! \param acceptTokenNum       [out] number of accepted draft tokens per batch [bs]
//! \param candidates           [in]  candidate token id per tree node [bs, numDraftTokens]
//! \param retrieveIndex        [in]  tree node -> local index [bs, numDraftTokens]
//! \param retrieveNextToken    [in]  first-child pointer [bs, numDraftTokens], -1=none
//! \param retrieveNextSibling  [in]  next-sibling pointer [bs, numDraftTokens], -1=none
//! \param targetPredict        [in]  target model prediction per position [bs * numDraftTokens]
//! \param batchSize            batch size
//! \param numSpeculativeTokens second dim of acceptIndex/acceptToken
//!                             (= numSpecStep = max_path_len, >= max possible accepts + 1)
//! \param numDraftTokens       total tree nodes per batch (including root)
__global__ void verifyDynamicTreeGreedyKernel(int64_t* predicts, int64_t* acceptIndex, int64_t* acceptTokenNum,
    int64_t* acceptToken, int64_t const* candidates, int32_t const* retrieveIndex, int32_t const* retrieveNextToken,
    int32_t const* retrieveNextSibling, int64_t const* targetPredict, bool const* treeValid, uint32_t batchSize,
    uint32_t numSpeculativeTokens, uint32_t numDraftTokens)
{
    uint32_t bx = blockIdx.x;
    uint32_t batchOffset = bx * numDraftTokens;

    // First-gen or dummy request: no valid tree, accept only the bonus token
    if (treeValid != nullptr && !treeValid[bx])
    {
        acceptTokenNum[bx] = 0;
        acceptIndex[bx * numSpeculativeTokens] = 0;
        acceptToken[bx * numSpeculativeTokens] = targetPredict[batchOffset];
        predicts[batchOffset] = targetPredict[batchOffset];
        return;
    }

    int32_t lastAcceptedLocalIdx = retrieveIndex[batchOffset];
    acceptIndex[bx * numSpeculativeTokens] = lastAcceptedLocalIdx;
    uint32_t numAcceptedTokens = 0;
    int32_t curIndex = 0;

    // Root token: target prediction at root position
    acceptToken[bx * numSpeculativeTokens] = targetPredict[batchOffset + lastAcceptedLocalIdx];

    for (uint32_t j = 1; j < numSpeculativeTokens; ++j)
    {
        curIndex = retrieveNextToken[batchOffset + curIndex];

        while (curIndex != -1)
        {
            int32_t draftLocalIdx = retrieveIndex[batchOffset + curIndex];
            int64_t draftTokenId = candidates[batchOffset + curIndex];
            int64_t targetTokenId = targetPredict[batchOffset + lastAcceptedLocalIdx];

            if (draftTokenId == targetTokenId)
            {
                predicts[batchOffset + lastAcceptedLocalIdx] = targetTokenId;
                ++numAcceptedTokens;
                acceptIndex[bx * numSpeculativeTokens + numAcceptedTokens] = draftLocalIdx;
                // Accepted token: target prediction at accepted draft position
                acceptToken[bx * numSpeculativeTokens + numAcceptedTokens] = targetPredict[batchOffset + draftLocalIdx];
                lastAcceptedLocalIdx = draftLocalIdx;
                break;
            }
            else
            {
                curIndex = retrieveNextSibling[batchOffset + curIndex];
            }
        }

        if (curIndex == -1)
            break;
    }

    acceptTokenNum[bx] = numAcceptedTokens;
    // Bonus token from target model at the last accepted position
    predicts[batchOffset + lastAcceptedLocalIdx] = targetPredict[batchOffset + lastAcceptedLocalIdx];
}

void invokeVerifyDynamicTreeGreedy(int64_t* predicts, int64_t* acceptIndex, int64_t* acceptTokenNum,
    int64_t* acceptToken, int64_t const* candidates, int32_t const* retrieveIndex, int32_t const* retrieveNextToken,
    int32_t const* retrieveNextSibling, int64_t const* targetPredict, bool const* treeValid, SizeType32 batchSize,
    SizeType32 numDraftTokens, SizeType32 numSpecStep, cudaStream_t stream)
{
    dim3 grid(batchSize);
    dim3 block(1);

    verifyDynamicTreeGreedyKernel<<<grid, block, 0, stream>>>(predicts, acceptIndex, acceptTokenNum, acceptToken,
        candidates, retrieveIndex, retrieveNextToken, retrieveNextSibling, targetPredict, treeValid, batchSize,
        numSpecStep, numDraftTokens);

    sync_check_cuda_error(stream);
}

// ------------------------------------------------------------
// Background: Speculative Sampling Theory
// ------------------------------------------------------------
//
// Goal: reuse draft model samples to speed up generation while keeping the
// final output distribution strictly equal to the target distribution q.
//
// For a given token x:
//   p(x) = draft_probs[x]   (draft model probability)
//   q(x) = target_probs[x]  (target model probability)
//
// Step 1 - The draft model proposes token x sampled from p.
// Step 2 - Accept x with probability min(1, q(x)/p(x)).
//          Equivalently: accept when u * p(x) < q(x), where u ~ Uniform(0,1).
//
// Why does this work?
//   x is proposed with probability p(x) and then accepted with probability
//   min(1, q(x)/p(x)), so its total probability mass reaching the output is:
//     p(x) * min(1, q(x)/p(x)) = min(p(x), q(x))
//
//   This covers only the min(p, q) portion of the target mass.
//   The remaining portion q - min(p, q) = relu(q - p) is not yet covered.
//
//   Therefore, if the draft token is rejected, we must resample from the
//   residual distribution relu(q - p) (normalised) to fill the gap and
//   restore the full target distribution.
//
// Example:
//   p = [0.6, 0.3, 0.1]   tokens [A, B, C]
//   q = [0.2, 0.5, 0.3]
//
//   Accept probabilities:
//     A: min(1, 0.2/0.6) = 1/3     B: min(1, 0.5/0.3) = 1     C: min(1, 0.3/0.1) = 1
//
//   Case 1 - draft proposes A (prob 0.6):
//     Accept (1/3):  contributes 0.6 * 1/3 = 0.2 to output A.
//     Reject (2/3):  total rejected mass = 0.6 * 2/3 = 0.4.
//       relu(q-p) = [0, 0.2, 0.2]  ->  normalised [0, 0.5, 0.5]
//       contributes 0.4*0.5 = 0.2 to B and 0.4*0.5 = 0.2 to C.
//   Case 2 - draft proposes B (prob 0.3): always accepted -> 0.3 to B.
//   Case 3 - draft proposes C (prob 0.1): always accepted -> 0.1 to C.
//
//   Final output distribution:
//     A = 0.2,  B = 0.3 + 0.2 = 0.5,  C = 0.1 + 0.2 = 0.3  ->  exactly q.
//
// Tree extension:
//   The same logic applies depth-by-depth along the draft tree. At each
//   depth the kernel tries siblings in score order; the first accepted
//   sibling extends the current path. If every sibling at a depth is
//   rejected the kernel samples a correction token from relu(q-p) and
//   terminates traversal for that request.
// ------------------------------------------------------------

#include <curand_kernel.h>

/// Map curand_uniform (0, 1] to [0, 1) so that cumulative-sum sampling
/// never falls off the end of a probability distribution due to float32
/// rounding.  1.0 is mapped to 0.0 (probability mass epsilon).
__device__ __forceinline__ float curand_uniform_open_right(curandStatePhilox4_32_10_t& state)
{
    float u = curand_uniform(&state); // (0, 1]
    return u < 1.0f ? u : 0.0f;       // [0, 1)
}

__device__ int64_t sampleFromDistribution(curandStatePhilox4_32_10_t& state, float const* probs, uint32_t vocabSize)
{
    float r = curand_uniform_open_right(state); // [0, 1)
    float cumsum = 0.0f;
    int64_t sampledTok = 0;

    for (uint32_t v = 0; v < vocabSize; ++v)
    {
        cumsum += probs[v];
        if (r < cumsum)
        {
            sampledTok = static_cast<int64_t>(v);
            return sampledTok;
        }
    }

    // Float32 cumsum may not reach 1.0 for large vocabs.
    // Fall back to the last token with positive probability.
    for (int64_t v = static_cast<int64_t>(vocabSize) - 1; v >= 0; --v)
    {
        if (probs[v] > 0.0f)
        {
            return v;
        }
    }
    return static_cast<int64_t>(vocabSize) - 1;
}

__device__ int64_t sampleFromIndexedDistribution(curandStatePhilox4_32_10_t& state, float const* probs,
    int32_t const* supportIndices, uint32_t supportSize, uint32_t vocabSize)
{
    float r = curand_uniform_open_right(state); // [0, 1)
    float cumsum = 0.0f;
    int64_t sampledTok = static_cast<int64_t>(vocabSize) - 1;

    for (uint32_t i = 0; i < supportSize; ++i)
    {
        int32_t const tok = supportIndices[i];
        cumsum += probs[tok];
        if (r < cumsum)
        {
            return static_cast<int64_t>(tok);
        }
    }

    // Fallback: last support token with positive probability.
    for (int64_t i = static_cast<int64_t>(supportSize) - 1; i >= 0; --i)
    {
        if (probs[supportIndices[i]] > 0.0f)
        {
            return static_cast<int64_t>(supportIndices[i]);
        }
    }
    return sampledTok;
}

struct MinInt32Op
{
    __device__ __forceinline__ int32_t operator()(int32_t a, int32_t b) const
    {
        return a < b ? a : b;
    }
};

struct MaxInt32Op
{
    __device__ __forceinline__ int32_t operator()(int32_t a, int32_t b) const
    {
        return a > b ? a : b;
    }
};

struct MaxFloatOp
{
    __device__ __forceinline__ float operator()(float a, float b) const
    {
        return a > b ? a : b;
    }
};

struct SoftmaxStats
{
    float maxVal;
    float sumVal;
    int32_t argmax;
};

//! \param acceptIndex          [out] accepted path as tree positions [bs, numSpecStep]. int64.
//! \param acceptTokenNum       [out] number of accepted draft tokens (excl. root) [bs]. int64.
//! \param acceptToken          [out] emitted token ids [bs, numSpecStep]. int64.
//! \param candidates           [in]  candidate token ids [bs, numDraftTokens]; col 0 = root. int64.
//! \param draftProbs           [in]  unique draft probs [bs, numDraftProbRows, vocabSize]. float32.
//! \param targetProbs          [in]  target probs [bs, numDraftTokens, vocabSize]; index 0 = root. float32.
//! \param targetSupportIndices [in] compact target support per tree position
//!                              [bs, numDraftTokens, maxTargetSupportSize]. int32, or nullptr.
//! \param targetSupportLengths [in] support length per tree position [bs, numDraftTokens]. int32, or nullptr.
//! \param draftProbIndices     [in]  tree position -> draftProbs row [bs, numDraftTokens], root unused. int32.
//! \param retrieveNextToken    [in]  first-child pointer [bs, numDraftTokens], -1=none. int32.
//! \param retrieveNextSibling  [in]  next-sibling pointer [bs, numDraftTokens], -1=none. int32.
//! \param treeValid            [in]  per-request tree validity flag [bs]. bool.
//! \param batchSize            batch size.
//! \param numDraftProbRows     unique draft-prob rows per request.
//! \param maxTargetSupportSize support-array width. Zero when targetSupportIndices is null.
//! \param numSpecStep          second dim of acceptIndex/acceptToken
//!                             (= max_path_len = max_draft_len + 1).
//! \param numDraftTokens       total tree nodes per batch (including root).
//! \param vocabSize            vocabulary size.
//! \param seed                 [1] int64 on GPU. Philox RNG seed.
//! \param offset               [1] int64 on GPU. Philox RNG offset.
template <int32_t BLOCK_SIZE, bool USE_LOGITS>
__global__ void verifyDynamicTreeRejectionKernel(int64_t* acceptIndex, int64_t* acceptTokenNum, int64_t* acceptToken,
    int64_t const* candidates, float const* draftInputs, float const* targetInputs, int32_t const* targetSupportIndices,
    int32_t const* targetSupportLengths, int32_t const* draftProbIndices, int32_t const* retrieveNextToken,
    int32_t const* retrieveNextSibling, bool const* treeValid, uint32_t batchSize, uint32_t numDraftProbRows,
    uint32_t maxTargetSupportSize, uint32_t numSpeculativeTokens, uint32_t numDraftTokens, uint32_t vocabSize,
    uint32_t draftVocabSize, int32_t const* targetToDraft, int64_t const* seed, int64_t const* offset,
    float const* temperatures)
{
    uint32_t bx = blockIdx.x;
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    constexpr uint32_t kVecSize = 4;
    if (bx >= batchSize)
    {
        return;
    }

    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    using BlockReduceInt = cub::BlockReduce<int32_t, BLOCK_SIZE>;
    using BlockScan = cub::BlockScan<float, BLOCK_SIZE>;

    __shared__ union
    {
        typename BlockReduce::TempStorage reduce;
        typename BlockReduceInt::TempStorage reduceInt;
        typename BlockScan::TempStorage scan;
    } tempStorage;

    __shared__ int32_t sLastAcceptedLocalIdx;
    __shared__ uint32_t sNumAcceptedTokens;
    __shared__ int32_t sFirstChild;
    __shared__ bool sHasTerminalToken;
    __shared__ float sDiffSum;
    __shared__ float sTargetMass;
    __shared__ float sPrefixBase;
    __shared__ int32_t sWinnerIndex;
    __shared__ int32_t sLastValidIndex;
    __shared__ int64_t sSampledToken;
    __shared__ float sLogitsMax;
    __shared__ float sLogitsSum;
    __shared__ int32_t sLogitsArgmax;

    // The first sibling that passes the rejection test at the current depth.
    __shared__ int32_t sAccSibIdx;
    __shared__ int64_t sAccSibTok;
    __shared__ int32_t sNumAccSiblings;

    uint32_t batchOffset = bx * numDraftTokens;

    curandStatePhilox4_32_10_t state;
    if (tid == 0)
    {
        curand_init(
            static_cast<uint64_t>(seed[0]), static_cast<uint64_t>(bx), static_cast<uint64_t>(offset[0]), &state);
    }
    __syncthreads();
    bool const hasCompactTargetSupport = targetSupportIndices != nullptr && targetSupportLengths != nullptr;
    bool const isGreedyRequest
        = USE_LOGITS && temperatures != nullptr && temperatures[bx] <= static_cast<float>(kGreedyTempThreshold);
    float const* draftProbs = draftInputs;
    float const* targetProbs = targetInputs;
    uint32_t const draftRowStride = USE_LOGITS ? draftVocabSize : vocabSize;

    auto canVectorizeLoad = [&](float const* probs, uint32_t rowSize) -> bool
    {
        constexpr uint32_t kLoadAlignmentBytes = kVecSize * sizeof(float);
        return (rowSize % kVecSize == 0) && (reinterpret_cast<std::uintptr_t>(probs) % kLoadAlignmentBytes == 0);
    };

    auto loadProbVec = [&](float const* probs, uint32_t base, bool useVectorizedLoads, uint32_t rowSize)
    {
        flashinfer::vec_t<float, kVecSize> probVec;
        probVec.fill(0.0f);
        if (useVectorizedLoads && base + kVecSize <= rowSize)
        {
            probVec.cast_load(probs + base);
        }
        else
        {
#pragma unroll
            for (uint32_t j = 0; j < kVecSize; ++j)
            {
                uint32_t const v = base + j;
                if (v < rowSize)
                {
                    probVec[j] = probs[v];
                }
            }
        }
        return probVec;
    };

    auto computeLogitsStats = [&](float const* logitsRow, uint32_t rowSize) -> SoftmaxStats
    {
        float threadMax = -FLT_MAX;
        for (uint32_t v = static_cast<uint32_t>(tid); v < rowSize; v += BLOCK_SIZE)
        {
            threadMax = fmaxf(threadMax, logitsRow[v]);
        }

        float const blockMax = BlockReduce(tempStorage.reduce).Reduce(threadMax, MaxFloatOp{});
        if (tid == 0)
        {
            sLogitsMax = blockMax;
        }
        __syncthreads();

        float threadSum = 0.0f;
        int32_t localArgmax = static_cast<int32_t>(rowSize);
        for (uint32_t v = static_cast<uint32_t>(tid); v < rowSize; v += BLOCK_SIZE)
        {
            float const logit = logitsRow[v];
            threadSum += __expf(logit - sLogitsMax);
            if (logit == sLogitsMax && localArgmax == static_cast<int32_t>(rowSize))
            {
                localArgmax = static_cast<int32_t>(v);
            }
        }

        float const blockSum = BlockReduce(tempStorage.reduce).Sum(threadSum);
        __syncthreads();
        int32_t const blockArgmax = BlockReduceInt(tempStorage.reduceInt).Reduce(localArgmax, MinInt32Op{});
        if (tid == 0)
        {
            sLogitsSum = blockSum;
            sLogitsArgmax = blockArgmax;
        }
        __syncthreads();

        return SoftmaxStats{sLogitsMax, sLogitsSum, sLogitsArgmax};
    };

    auto probFromLogits = [&](float const* logitsRow, uint32_t tokenId, uint32_t rowSize, SoftmaxStats stats) -> float
    {
        if (tokenId >= rowSize)
        {
            return 0.0f;
        }
        if (isGreedyRequest)
        {
            return tokenId == static_cast<uint32_t>(stats.argmax) ? 1.0f : 0.0f;
        }
        constexpr float kFloatSoftmaxEpsilon = 1e-6f;
        return __expf(logitsRow[tokenId] - stats.maxVal) / (stats.sumVal + kFloatSoftmaxEpsilon);
    };

    auto targetTokenToDraftToken = [&](uint32_t targetTokenId) -> int32_t
    {
        if (targetTokenId >= vocabSize)
        {
            return -1;
        }
        if (targetToDraft != nullptr)
        {
            return targetToDraft[targetTokenId];
        }
        return targetTokenId < draftVocabSize ? static_cast<int32_t>(targetTokenId) : -1;
    };

    auto draftProbFromTargetToken
        = [&](float const* draftLogitsRow, uint32_t targetTokenId, SoftmaxStats stats) -> float
    {
        int32_t const draftTokenId = targetTokenToDraftToken(targetTokenId);
        if (draftTokenId < 0)
        {
            return 0.0f;
        }
        return probFromLogits(draftLogitsRow, static_cast<uint32_t>(draftTokenId), draftVocabSize, stats);
    };

    auto sampleProbTile = [&](float(&value)[kVecSize], uint32_t base) -> bool
    {
        float const tileSum = BlockReduce(tempStorage.reduce).template Sum<kVecSize>(value);
        if (tid == 0)
        {
            sDiffSum = tileSum;
        }
        __syncthreads();

        int32_t localLastValid = -1;
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j)
        {
            uint32_t const v = base + j;
            if (v < vocabSize && value[j] > 0.0f)
            {
                localLastValid = static_cast<int32_t>(v);
            }
        }
        int32_t const blockLastValid = BlockReduceInt(tempStorage.reduceInt).Reduce(localLastValid, MaxInt32Op{});
        if (tid == 0 && blockLastValid >= 0)
        {
            sLastValidIndex = blockLastValid;
        }
        __syncthreads();

        if (sPrefixBase + sDiffSum > sTargetMass)
        {
            float inclusive[kVecSize];
            BlockScan(tempStorage.scan).template InclusiveSum<kVecSize>(value, inclusive);
            __syncthreads();

            int32_t localWinner = static_cast<int32_t>(vocabSize);
#pragma unroll
            for (uint32_t j = 0; j < kVecSize; ++j)
            {
                uint32_t const v = base + j;
                if (v < vocabSize && value[j] > 0.0f && sPrefixBase + inclusive[j] > sTargetMass)
                {
                    localWinner = static_cast<int32_t>(v);
                    break;
                }
            }

            int32_t const blockWinner = BlockReduceInt(tempStorage.reduceInt).Reduce(localWinner, MinInt32Op{});
            if (tid == 0)
            {
                sWinnerIndex = blockWinner;
                sSampledToken = blockWinner < static_cast<int32_t>(vocabSize) ? static_cast<int64_t>(blockWinner)
                                                                              : static_cast<int64_t>(vocabSize) - 1;
            }
            __syncthreads();
            return true;
        }

        if (tid == 0)
        {
            sPrefixBase += sDiffSum;
        }
        __syncthreads();
        return false;
    };

    auto sampleTargetFullVocab = [&](float const* tProbs)
    {
        bool const useVectorizedLoads = canVectorizeLoad(tProbs, vocabSize);
        uint32_t const numIters = (vocabSize + BLOCK_SIZE * kVecSize - 1) / (BLOCK_SIZE * kVecSize);
        if (tid == 0)
        {
            sPrefixBase = 0.0f;
            sWinnerIndex = static_cast<int32_t>(vocabSize);
            sLastValidIndex = -1;
            sSampledToken = static_cast<int64_t>(vocabSize) - 1;
            sTargetMass = curand_uniform_open_right(state);
        }
        __syncthreads();

#pragma unroll 2
        for (uint32_t i = 0; i < numIters; ++i)
        {
            uint32_t const base = (i * BLOCK_SIZE + static_cast<uint32_t>(tid)) * kVecSize;
            auto const qVec = loadProbVec(tProbs, base, useVectorizedLoads, vocabSize);
            float value[kVecSize];
#pragma unroll
            for (uint32_t j = 0; j < kVecSize; ++j)
            {
                value[j] = qVec[j];
            }

            if (sampleProbTile(value, base))
            {
                break;
            }
        }

        if (tid == 0 && sWinnerIndex >= static_cast<int32_t>(vocabSize) && sLastValidIndex >= 0)
        {
            sSampledToken = static_cast<int64_t>(sLastValidIndex);
        }
        __syncthreads();
    };

    auto sampleResidualFullVocab = [&](float const* tProbs, float const* dProbs)
    {
        bool const useVectorizedTargetLoads = canVectorizeLoad(tProbs, vocabSize);
        bool const useVectorizedDraftLoads = canVectorizeLoad(dProbs, vocabSize);
        uint32_t const numIters = (vocabSize + BLOCK_SIZE * kVecSize - 1) / (BLOCK_SIZE * kVecSize);
        float totalSum = 0.0f;
#pragma unroll 2
        for (uint32_t i = 0; i < numIters; ++i)
        {
            uint32_t const base = (i * BLOCK_SIZE + static_cast<uint32_t>(tid)) * kVecSize;
            auto const qVec = loadProbVec(tProbs, base, useVectorizedTargetLoads, vocabSize);
            auto const pVec = loadProbVec(dProbs, base, useVectorizedDraftLoads, vocabSize);
            float value[kVecSize];
#pragma unroll
            for (uint32_t j = 0; j < kVecSize; ++j)
            {
                value[j] = fmaxf(qVec[j] - pVec[j], 0.0f);
            }
            totalSum += BlockReduce(tempStorage.reduce).template Sum<kVecSize>(value);
            __syncthreads();
        }

        if (tid == 0)
        {
            sDiffSum = totalSum;
            sPrefixBase = 0.0f;
            sWinnerIndex = static_cast<int32_t>(vocabSize);
            sLastValidIndex = -1;
            sSampledToken = static_cast<int64_t>(vocabSize) - 1;
            sTargetMass = totalSum > 1e-10f ? curand_uniform_open_right(state) * totalSum : 0.0f;
            if (totalSum <= 1e-10f)
            {
                sSampledToken = sampleFromDistribution(state, tProbs, vocabSize);
            }
        }
        __syncthreads();

        if (sDiffSum <= 1e-10f)
        {
            return;
        }

#pragma unroll 2
        for (uint32_t i = 0; i < numIters; ++i)
        {
            uint32_t const base = (i * BLOCK_SIZE + static_cast<uint32_t>(tid)) * kVecSize;
            auto const qVec = loadProbVec(tProbs, base, useVectorizedTargetLoads, vocabSize);
            auto const pVec = loadProbVec(dProbs, base, useVectorizedDraftLoads, vocabSize);
            float value[kVecSize];
#pragma unroll
            for (uint32_t j = 0; j < kVecSize; ++j)
            {
                value[j] = fmaxf(qVec[j] - pVec[j], 0.0f);
            }

            if (sampleProbTile(value, base))
            {
                break;
            }
        }

        if (tid == 0 && sWinnerIndex >= static_cast<int32_t>(vocabSize) && sLastValidIndex >= 0)
        {
            sSampledToken = static_cast<int64_t>(sLastValidIndex);
        }
        __syncthreads();
    };

    auto sampleTargetLogitsFullVocabWithStats = [&](float const* tLogits, SoftmaxStats targetStats)
    {
        if (tid == 0)
        {
            constexpr float kFloatSoftmaxEpsilon = 1e-6f;
            sPrefixBase = 0.0f;
            sWinnerIndex = static_cast<int32_t>(vocabSize);
            sLastValidIndex = -1;
            sSampledToken = static_cast<int64_t>(vocabSize) - 1;
            sTargetMass = isGreedyRequest
                ? 0.0f
                : curand_uniform_open_right(state) * (targetStats.sumVal + kFloatSoftmaxEpsilon);
            if (isGreedyRequest)
            {
                sSampledToken = static_cast<int64_t>(targetStats.argmax);
            }
        }
        __syncthreads();

        if (isGreedyRequest)
        {
            return;
        }

        bool const useVectorizedLoads = canVectorizeLoad(tLogits, vocabSize);
        uint32_t const numIters = (vocabSize + BLOCK_SIZE * kVecSize - 1) / (BLOCK_SIZE * kVecSize);
#pragma unroll 2
        for (uint32_t i = 0; i < numIters; ++i)
        {
            uint32_t const base = (i * BLOCK_SIZE + static_cast<uint32_t>(tid)) * kVecSize;
            auto const qVec = loadProbVec(tLogits, base, useVectorizedLoads, vocabSize);
            float value[kVecSize];
#pragma unroll
            for (uint32_t j = 0; j < kVecSize; ++j)
            {
                uint32_t const v = base + j;
                value[j] = v < vocabSize ? __expf(qVec[j] - targetStats.maxVal) : 0.0f;
            }

            if (sampleProbTile(value, base))
            {
                break;
            }
        }

        if (tid == 0 && sWinnerIndex >= static_cast<int32_t>(vocabSize) && sLastValidIndex >= 0)
        {
            sSampledToken = static_cast<int64_t>(sLastValidIndex);
        }
        __syncthreads();
    };

    auto sampleTargetLogitsFullVocab = [&](float const* tLogits)
    {
        auto const targetStats = computeLogitsStats(tLogits, vocabSize);
        sampleTargetLogitsFullVocabWithStats(tLogits, targetStats);
    };

    auto sampleResidualLogitsFullVocab
        = [&](float const* tLogits, float const* dLogits, SoftmaxStats targetStats, SoftmaxStats draftStats)
    {
        if (isGreedyRequest)
        {
            if (tid == 0)
            {
                sSampledToken = static_cast<int64_t>(targetStats.argmax);
            }
            __syncthreads();
            return;
        }

        bool const useVectorizedTargetLoads = canVectorizeLoad(tLogits, vocabSize);
        bool const useVectorizedDraftLoads = targetToDraft == nullptr && canVectorizeLoad(dLogits, draftVocabSize);
        uint32_t const numIters = (vocabSize + BLOCK_SIZE * kVecSize - 1) / (BLOCK_SIZE * kVecSize);
        constexpr float kFloatSoftmaxEpsilon = 1e-6f;
        float totalSum = 0.0f;
#pragma unroll 2
        for (uint32_t i = 0; i < numIters; ++i)
        {
            uint32_t const base = (i * BLOCK_SIZE + static_cast<uint32_t>(tid)) * kVecSize;
            auto const qVec = loadProbVec(tLogits, base, useVectorizedTargetLoads, vocabSize);
            flashinfer::vec_t<float, kVecSize> pVec;
            pVec.fill(0.0f);
            if (targetToDraft == nullptr)
            {
                pVec = loadProbVec(dLogits, base, useVectorizedDraftLoads, draftVocabSize);
            }
            float value[kVecSize];
#pragma unroll
            for (uint32_t j = 0; j < kVecSize; ++j)
            {
                uint32_t const v = base + j;
                if (v < vocabSize)
                {
                    float const q = __expf(qVec[j] - targetStats.maxVal) / (targetStats.sumVal + kFloatSoftmaxEpsilon);
                    int32_t const draftTokenId = targetTokenToDraftToken(v);
                    float p = 0.0f;
                    if (draftTokenId >= 0 && draftTokenId < static_cast<int32_t>(draftVocabSize))
                    {
                        float const draftLogit = targetToDraft == nullptr ? pVec[j] : dLogits[draftTokenId];
                        p = __expf(draftLogit - draftStats.maxVal) / (draftStats.sumVal + kFloatSoftmaxEpsilon);
                    }
                    value[j] = fmaxf(q - p, 0.0f);
                }
                else
                {
                    value[j] = 0.0f;
                }
            }
            totalSum += BlockReduce(tempStorage.reduce).template Sum<kVecSize>(value);
            __syncthreads();
        }

        if (tid == 0)
        {
            sDiffSum = totalSum;
            sPrefixBase = 0.0f;
            sWinnerIndex = static_cast<int32_t>(vocabSize);
            sLastValidIndex = -1;
            sSampledToken = static_cast<int64_t>(vocabSize) - 1;
            sTargetMass = totalSum > 1e-10f ? curand_uniform_open_right(state) * totalSum : 0.0f;
        }
        __syncthreads();

        if (sDiffSum <= 1e-10f)
        {
            sampleTargetLogitsFullVocabWithStats(tLogits, targetStats);
            return;
        }

#pragma unroll 2
        for (uint32_t i = 0; i < numIters; ++i)
        {
            uint32_t const base = (i * BLOCK_SIZE + static_cast<uint32_t>(tid)) * kVecSize;
            auto const qVec = loadProbVec(tLogits, base, useVectorizedTargetLoads, vocabSize);
            flashinfer::vec_t<float, kVecSize> pVec;
            pVec.fill(0.0f);
            if (targetToDraft == nullptr)
            {
                pVec = loadProbVec(dLogits, base, useVectorizedDraftLoads, draftVocabSize);
            }
            float value[kVecSize];
#pragma unroll
            for (uint32_t j = 0; j < kVecSize; ++j)
            {
                uint32_t const v = base + j;
                if (v < vocabSize)
                {
                    float const q = __expf(qVec[j] - targetStats.maxVal) / (targetStats.sumVal + kFloatSoftmaxEpsilon);
                    int32_t const draftTokenId = targetTokenToDraftToken(v);
                    float p = 0.0f;
                    if (draftTokenId >= 0 && draftTokenId < static_cast<int32_t>(draftVocabSize))
                    {
                        float const draftLogit = targetToDraft == nullptr ? pVec[j] : dLogits[draftTokenId];
                        p = __expf(draftLogit - draftStats.maxVal) / (draftStats.sumVal + kFloatSoftmaxEpsilon);
                    }
                    value[j] = fmaxf(q - p, 0.0f);
                }
                else
                {
                    value[j] = 0.0f;
                }
            }

            if (sampleProbTile(value, base))
            {
                break;
            }
        }

        if (tid == 0 && sWinnerIndex >= static_cast<int32_t>(vocabSize) && sLastValidIndex >= 0)
        {
            sSampledToken = static_cast<int64_t>(sLastValidIndex);
        }
        __syncthreads();
    };

    // First-gen or dummy request: no valid tree exists yet. Sample directly
    // from the target distribution at the root and skip tree traversal.
    if (treeValid != nullptr && !treeValid[bx])
    {
        float const* tProbs = targetProbs + static_cast<uint64_t>(bx) * numDraftTokens * vocabSize;
        if (hasCompactTargetSupport)
        {
            if (tid == 0)
            {
                uint32_t const supportOffset = static_cast<uint64_t>(bx) * numDraftTokens * maxTargetSupportSize;
                uint32_t const supportSize = static_cast<uint32_t>(targetSupportLengths[batchOffset]);
                sSampledToken = sampleFromIndexedDistribution(
                    state, tProbs, targetSupportIndices + supportOffset, supportSize, vocabSize);
            }
        }
        else
        {
            if constexpr (USE_LOGITS)
            {
                sampleTargetLogitsFullVocab(tProbs);
            }
            else
            {
                sampleTargetFullVocab(tProbs);
            }
        }
        if (tid == 0)
        {
            acceptIndex[bx * numSpeculativeTokens] = 0;
            acceptTokenNum[bx] = 0;
            acceptToken[bx * numSpeculativeTokens] = sSampledToken;
        }
        return;
    }

    // Root (depth 0): initialize path state at tree position 0.
    //
    // Example tree used in code review discussions:
    //   root: E
    //   children of E:   F1, F2, F3
    //   children of F1:  G1, G2
    //   children of F2:  G3
    //
    // In that example the per-request inputs are conceptually:
    //   candidates   = [E, F1, F2, F3, G1, G2, G3]
    //   draftProbs   = [p(.|E), p(.|F1), p(.|F2)]
    //   draftProbIndices = [0, 0, 0, 0, 1, 1, 2]
    //   targetProbs  = [q(.|E), q(.|F1), q(.|F2), q(.|F3), q(.|G1), q(.|G2), q(.|G3)]
    //
    // draftProbs stores one row per unique parent context, so siblings that share
    // the same parent also share the same draftProbs row via draftProbIndices.
    // targetProbs remains aligned to all tree positions, including the root at slot 0.
    //
    // Output convention:
    //   - acceptIndex stores the accepted draft path as tree positions, with slot 0
    //     reserved for the root position.
    //   - acceptToken stores the emitted token sequence, matching the greedy kernel:
    //       slot 0                  = first emitted token
    //       slot numAcceptedTokens  = final bonus/correction token
    //   - acceptTokenNum stores the number of accepted draft tokens only. The caller
    //     adds 1 to obtain the total number of emitted tokens.
    if (tid == 0)
    {
        sLastAcceptedLocalIdx = 0;
        acceptIndex[bx * numSpeculativeTokens] = sLastAcceptedLocalIdx;
        sNumAcceptedTokens = 0;
        sHasTerminalToken = false;
    }
    __syncthreads();
    for (uint32_t j = 1; j < numSpeculativeTokens; ++j)
    {
        // Get first child of the last accepted node.
        if (tid == 0)
        {
            sFirstChild = retrieveNextToken[batchOffset + sLastAcceptedLocalIdx];
        }
        __syncthreads();

        // Leaf node: no children at this depth.
        // Emit bonus token from the target distribution at the last accepted position.
        if (sFirstChild == -1)
        {
            float const* tProbs
                = targetProbs + (static_cast<uint64_t>(bx) * numDraftTokens + sLastAcceptedLocalIdx) * vocabSize;
            if (hasCompactTargetSupport)
            {
                if (tid == 0)
                {
                    uint32_t const supportOffset
                        = (static_cast<uint64_t>(bx) * numDraftTokens + sLastAcceptedLocalIdx) * maxTargetSupportSize;
                    uint32_t const supportSize
                        = static_cast<uint32_t>(targetSupportLengths[batchOffset + sLastAcceptedLocalIdx]);
                    sSampledToken = sampleFromIndexedDistribution(
                        state, tProbs, targetSupportIndices + supportOffset, supportSize, vocabSize);
                }
            }
            else
            {
                if constexpr (USE_LOGITS)
                {
                    sampleTargetLogitsFullVocab(tProbs);
                }
                else
                {
                    sampleTargetFullVocab(tProbs);
                }
            }
            if (tid == 0)
            {
                acceptToken[bx * numSpeculativeTokens + sNumAcceptedTokens] = sSampledToken;
                sHasTerminalToken = true;
            }
            __syncthreads();
            break;
        }

        // Test siblings in linked-list order. Once a sibling passes the
        // Bernoulli rejection test, accept it immediately and skip the rest.
        int32_t const firstDraftProbRow = draftProbIndices[batchOffset + sFirstChild];
        float const* siblingTargetRow
            = targetProbs + (static_cast<uint64_t>(bx) * numDraftTokens + sLastAcceptedLocalIdx) * vocabSize;
        float const* siblingDraftRow
            = draftProbs + (static_cast<uint64_t>(bx) * numDraftProbRows + firstDraftProbRow) * draftRowStride;
        SoftmaxStats siblingTargetStats{};
        SoftmaxStats siblingDraftStats{};
        if constexpr (USE_LOGITS)
        {
            siblingTargetStats = computeLogitsStats(siblingTargetRow, vocabSize);
            siblingDraftStats = computeLogitsStats(siblingDraftRow, draftVocabSize);
        }

        if (tid == 0)
        {
            sNumAccSiblings = 0;
            int32_t childIdx = sFirstChild;
            while (childIdx != -1)
            {
                int64_t const draftTokenId = candidates[batchOffset + childIdx];
                int32_t const draftProbRow = draftProbIndices[batchOffset + childIdx];
                uint32_t const tokenId = static_cast<uint32_t>(draftTokenId);
                float const pDraft = USE_LOGITS
                    ? draftProbFromTargetToken(siblingDraftRow, tokenId, siblingDraftStats)
                    : draftProbs[(static_cast<uint64_t>(bx) * numDraftProbRows + draftProbRow) * vocabSize
                        + draftTokenId];
                float const pTarget = USE_LOGITS
                    ? probFromLogits(siblingTargetRow, tokenId, vocabSize, siblingTargetStats)
                    : targetProbs[(static_cast<uint64_t>(bx) * numDraftTokens + sLastAcceptedLocalIdx) * vocabSize
                        + draftTokenId];

                float const acceptProb = fminf(1.0f, pTarget / (pDraft + 1e-10f));
                float const u = curand_uniform_open_right(state);

                if (u < acceptProb)
                {
                    sAccSibIdx = childIdx;
                    sAccSibTok = draftTokenId;
                    sNumAccSiblings = 1;
                    break;
                }
                childIdx = retrieveNextSibling[batchOffset + childIdx];
            }
        }
        __syncthreads();

        // Select the first accepted sibling or emit correction when all siblings reject.
        if (sNumAccSiblings > 0)
        {
            if (tid == 0)
            {
                int32_t const childIdx = sAccSibIdx;
                acceptToken[bx * numSpeculativeTokens + sNumAcceptedTokens] = sAccSibTok;
                ++sNumAcceptedTokens;
                acceptIndex[bx * numSpeculativeTokens + sNumAcceptedTokens] = childIdx;
                sLastAcceptedLocalIdx = childIdx;
            }
            __syncthreads();
        }
        else
        {
            // All siblings rejected -> sample correction token from relu(q - p).
            {
                int32_t const draftProbRow = draftProbIndices[batchOffset + sFirstChild];
                float const* tProbs
                    = targetProbs + (static_cast<uint64_t>(bx) * numDraftTokens + sLastAcceptedLocalIdx) * vocabSize;
                float const* dProbs
                    = draftProbs + (static_cast<uint64_t>(bx) * numDraftProbRows + draftProbRow) * draftRowStride;
                int32_t const* tProbIndices = nullptr;
                uint32_t targetSupportSize = vocabSize;
                if (hasCompactTargetSupport)
                {
                    uint32_t const supportOffset
                        = (static_cast<uint64_t>(bx) * numDraftTokens + sLastAcceptedLocalIdx) * maxTargetSupportSize;
                    tProbIndices = targetSupportIndices + supportOffset;
                    targetSupportSize
                        = static_cast<uint32_t>(targetSupportLengths[batchOffset + sLastAcceptedLocalIdx]);
                }

                if (hasCompactTargetSupport)
                {
                    if (tid == 0)
                    {
                        float diffSum = 0.0f;
                        for (uint32_t i = 0; i < targetSupportSize; ++i)
                        {
                            uint32_t const v = static_cast<uint32_t>(tProbIndices[i]);
                            float const diff = tProbs[v] - dProbs[v];
                            if (diff > 0.0f)
                            {
                                diffSum += diff;
                            }
                        }

                        int64_t corrTok = static_cast<int64_t>(vocabSize) - 1;
                        bool const useDiff = (diffSum > 1e-10f);

                        if (useDiff)
                        {
                            float const r = curand_uniform_open_right(state);
                            float cumsum = 0.0f;
                            for (uint32_t i = 0; i < targetSupportSize; ++i)
                            {
                                uint32_t const v = static_cast<uint32_t>(tProbIndices[i]);
                                float const diff = tProbs[v] - dProbs[v];
                                float const prob = (diff > 0.0f) ? diff / diffSum : 0.0f;
                                cumsum += prob;
                                if (r <= cumsum)
                                {
                                    corrTok = static_cast<int64_t>(v);
                                    break;
                                }
                            }
                        }
                        else
                        {
                            corrTok = sampleFromIndexedDistribution(
                                state, tProbs, tProbIndices, targetSupportSize, vocabSize);
                        }
                        acceptToken[bx * numSpeculativeTokens + sNumAcceptedTokens] = corrTok;
                        sHasTerminalToken = true;
                    }
                }
                else
                {
                    if constexpr (USE_LOGITS)
                    {
                        sampleResidualLogitsFullVocab(tProbs, dProbs, siblingTargetStats, siblingDraftStats);
                    }
                    else
                    {
                        sampleResidualFullVocab(tProbs, dProbs);
                    }

                    if (tid == 0)
                    {
                        acceptToken[bx * numSpeculativeTokens + sNumAcceptedTokens] = sSampledToken;
                        sHasTerminalToken = true;
                    }
                }
            }
            __syncthreads();
            break;
        }
    }

    if (!sHasTerminalToken)
    {
        // Reached max speculative depth while continuing to accept the draft path.
        // Emit the final bonus token from the last accepted position.
        float const* tProbs
            = targetProbs + (static_cast<uint64_t>(bx) * numDraftTokens + sLastAcceptedLocalIdx) * vocabSize;
        if (hasCompactTargetSupport)
        {
            if (tid == 0)
            {
                uint32_t const supportOffset
                    = (static_cast<uint64_t>(bx) * numDraftTokens + sLastAcceptedLocalIdx) * maxTargetSupportSize;
                uint32_t const supportSize
                    = static_cast<uint32_t>(targetSupportLengths[batchOffset + sLastAcceptedLocalIdx]);
                sSampledToken = sampleFromIndexedDistribution(
                    state, tProbs, targetSupportIndices + supportOffset, supportSize, vocabSize);
            }
        }
        else
        {
            if constexpr (USE_LOGITS)
            {
                sampleTargetLogitsFullVocab(tProbs);
            }
            else
            {
                sampleTargetFullVocab(tProbs);
            }
        }
        if (tid == 0)
        {
            acceptToken[bx * numSpeculativeTokens + sNumAcceptedTokens] = sSampledToken;
        }
    }

    if (tid == 0)
    {
        acceptTokenNum[bx] = sNumAcceptedTokens;
    }
}

void invokeVerifyDynamicTreeRejection(int64_t* acceptIndex, int64_t* acceptTokenNum, int64_t* acceptToken,
    int64_t const* candidates, float const* draftProbs, float const* targetProbs, int32_t const* targetSupportIndices,
    int32_t const* targetSupportLengths, int32_t const* draftProbIndices, int32_t const* retrieveNextToken,
    int32_t const* retrieveNextSibling, bool const* treeValid, SizeType32 batchSize, SizeType32 numDraftProbRows,
    SizeType32 maxTargetSupportSize, SizeType32 numDraftTokens, SizeType32 numSpecStep, SizeType32 vocabSize,
    int64_t const* seed, int64_t const* offset, cudaStream_t stream)
{
    constexpr int32_t kVerifyDynamicTreeRejectionBlockSize = 1024;
    dim3 grid(batchSize);
    dim3 block(kVerifyDynamicTreeRejectionBlockSize);

    verifyDynamicTreeRejectionKernel<kVerifyDynamicTreeRejectionBlockSize, false><<<grid, block, 0, stream>>>(
        acceptIndex, acceptTokenNum, acceptToken, candidates, draftProbs, targetProbs, targetSupportIndices,
        targetSupportLengths, draftProbIndices, retrieveNextToken, retrieveNextSibling, treeValid, batchSize,
        numDraftProbRows, maxTargetSupportSize, numSpecStep, numDraftTokens, vocabSize, vocabSize,
        /*targetToDraft=*/nullptr, seed, offset, nullptr);

    sync_check_cuda_error(stream);
}

} // namespace kernels::speculative_decoding

TRTLLM_NAMESPACE_END
