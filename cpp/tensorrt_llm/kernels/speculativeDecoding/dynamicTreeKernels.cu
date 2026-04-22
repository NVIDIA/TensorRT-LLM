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
#include "tensorrt_llm/kernels/decodingCommon.h"
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
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
    bool const hasTopK = topK.has_value() && topK->defined();
    bool const hasTopP = topP.has_value() && topP->defined();

    if (!hasTopK && !hasTopP)
    {
        return logits;
    }

    int64_t const vocabSize = logits.size(1);

    if (hasTopK && kMax > 0 && kMax < vocabSize)
    {
        // Fast topk path ─────────────────────────────────────────────────────────────
        // topKValues/topKIdx: [nRows, kMax], values in descending order
        auto [topKValues, topKIdx] = logits.topk(kMax, /*dim=*/-1, /*largest=*/true, /*sorted=*/true);

        // validTopK[i, j]: True when position j falls within top-K[i] for row i
        auto kArange = torch::arange(kMax, torch::TensorOptions().dtype(torch::kInt64).device(logits.device()))
                           .unsqueeze(0);                  // [1, kMax]
        auto kVals = topK->to(torch::kInt64).unsqueeze(1); // [nRows, 1]
        auto validTopK = kArange < kVals;                  // [nRows, kMax]

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
        auto topKMask = logitsSort.size(1) - topK->to(torch::kLong);
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

    auto scaledLogits
        = (skipTemperature ? logits : logits.div(temperatures.unsqueeze(1))).contiguous().to(torch::kFloat32);

    bool const hasTopK = topK.has_value() && topK->defined();
    int64_t const vocabSize = scaledLogits.size(1);
    int64_t const nRows = scaledLogits.size(0);

    torch::Tensor maskedLogits;
    if (hasTopK && kMax > 0 && kMax < vocabSize)
    {
        // Two-stage CUDA top-k/top-p masking (mirrors invokeBatchTopKSampling).
        maskedLogits = torch::empty_like(scaledLogits);
        auto stream = at::cuda::getCurrentCUDAStream(scaledLogits.device().index());
        invokeTopKTopPMaskingForProbs<float>(scaledLogits.data_ptr<float>(), maskedLogits.data_ptr<float>(),
            topK->to(torch::kInt32).contiguous().data_ptr<int32_t>(),
            (topP.has_value() && topP->defined()) ? topP->to(torch::kFloat32).contiguous().data_ptr<float>() : nullptr,
            kMax, static_cast<int32_t>(nRows), static_cast<int32_t>(vocabSize), stream);
    }
    else
    {
        // Fallback: PyTorch-based sort path (top-P only or kMax == 0).
        maskedLogits = applyTopKTopPForProbOp(scaledLogits, topK, topP, kMax);
    }

    auto probs = computeSoftmaxForProbOp(maskedLogits);

    auto isGreedy = temperatures <= kGreedyTempThreshold;
    auto argmaxIds = maskedLogits.argmax(/*dim=*/-1, /*keepdim=*/true);
    auto oneHot = torch::zeros_like(probs).scatter_(1, argmaxIds, 1.0);
    return torch::where(isGreedy.unsqueeze(1), oneHot, probs);
}

torch::Tensor computeDraftProbsForDynamicTreeRejection(torch::Tensor const& draftLogits,
    torch::Tensor const& temperatures, SizeType32 const numDraftProbRows, torch::optional<torch::Tensor> const& topK,
    torch::optional<torch::Tensor> const& topP, SizeType32 const targetVocabSize, bool skipTemperature,
    torch::optional<torch::Tensor> const& d2t, SizeType32 const kMax)
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

    auto draftTemps = temperatures.repeat_interleave(numDraftProbRows);
    auto draftTopK = topK.has_value() && topK->defined()
        ? torch::optional<torch::Tensor>(topK->repeat_interleave(numDraftProbRows))
        : torch::optional<torch::Tensor>();
    auto draftTopP = topP.has_value() && topP->defined()
        ? torch::optional<torch::Tensor>(topP->repeat_interleave(numDraftProbRows))
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
    SizeType32 const kMax)
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

    auto targetTemps = temperatures.repeat_interleave(numDraftTokens);
    auto targetTopK = topK.has_value() && topK->defined()
        ? torch::optional<torch::Tensor>(topK->repeat_interleave(numDraftTokens))
        : torch::optional<torch::Tensor>();
    auto targetTopP = topP.has_value() && topP->defined()
        ? torch::optional<torch::Tensor>(topP->repeat_interleave(numDraftTokens))
        : torch::optional<torch::Tensor>();

    bool const hasTopK = targetTopK.has_value() && targetTopK->defined();
    bool const hasTopP = targetTopP.has_value() && targetTopP->defined();
    bool const hasFiltering = hasTopK || hasTopP;

    auto scaledTargetLogits = (skipTemperature ? targetLogits : targetLogits.div(targetTemps.unsqueeze(1)))
                                  .contiguous()
                                  .to(torch::kFloat32);

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
    else if (hasTopK && kMax > 0 && static_cast<int64_t>(kMax) < targetVocabSize)
    {
        // Fast two-stage CUDA path for masked logits.
        maskedTargetLogits = torch::empty_like(scaledTargetLogits);
        auto stream = at::cuda::getCurrentCUDAStream(scaledTargetLogits.device().index());
        invokeTopKTopPMaskingForProbs<float>(scaledTargetLogits.data_ptr<float>(), maskedTargetLogits.data_ptr<float>(),
            targetTopK->to(torch::kInt32).contiguous().data_ptr<int32_t>(),
            hasTopP ? targetTopP->to(torch::kFloat32).contiguous().data_ptr<float>() : nullptr, kMax,
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
            auto topKMask = logitsSort.size(1) - targetTopK->to(torch::kLong);
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
        if (hasTopK && targetTopK->defined())
        {
            maxSupportSize = std::min<int64_t>(targetVocabSize, targetTopK->max().item<int64_t>());
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

    auto isGreedy = targetTemps <= kGreedyTempThreshold;
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

__device__ int64_t sampleFromDistribution(curandStatePhilox4_32_10_t& state, float const* probs, uint32_t vocabSize)
{
    float r = curand_uniform(&state);
    float cumsum = 0.0f;
    int64_t sampledTok = static_cast<int64_t>(vocabSize) - 1; // fallback: last vocab token

    for (uint32_t v = 0; v < vocabSize; ++v)
    {
        cumsum += probs[v];
        if (r <= cumsum)
        {
            sampledTok = static_cast<int64_t>(v);
            break;
        }
    }

    return sampledTok;
}

__device__ int64_t sampleFromIndexedDistribution(curandStatePhilox4_32_10_t& state, float const* probs,
    int32_t const* supportIndices, uint32_t supportSize, uint32_t vocabSize)
{
    float r = curand_uniform(&state);
    float cumsum = 0.0f;
    int64_t sampledTok = static_cast<int64_t>(vocabSize) - 1; // fallback: last vocab token

    for (uint32_t i = 0; i < supportSize; ++i)
    {
        int32_t const tok = supportIndices[i];
        cumsum += probs[tok];
        if (r <= cumsum)
        {
            sampledTok = static_cast<int64_t>(tok);
            break;
        }
    }

    return sampledTok;
}

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
//! \param batchSize            batch size.
//! \param numDraftProbRows     unique draft-prob rows per request.
//! \param maxTargetSupportSize support-array width. Zero when targetSupportIndices is null.
//! \param numSpecStep          second dim of acceptIndex/acceptToken
//!                             (= max_path_len = max_draft_len + 1).
//! \param numDraftTokens       total tree nodes per batch (including root).
//! \param vocabSize            vocabulary size.
//! \param seed                 [1] int64 on GPU. Philox RNG seed.
//! \param offset               [1] int64 on GPU. Philox RNG offset.
template <int32_t BLOCK_SIZE>
__global__ void verifyDynamicTreeRejectionKernel(int64_t* acceptIndex, int64_t* acceptTokenNum, int64_t* acceptToken,
    int64_t const* candidates, float const* draftProbs, float const* targetProbs, int32_t const* targetSupportIndices,
    int32_t const* targetSupportLengths, int32_t const* draftProbIndices, int32_t const* retrieveNextToken,
    int32_t const* retrieveNextSibling, uint32_t batchSize, uint32_t numDraftProbRows, uint32_t maxTargetSupportSize,
    uint32_t numSpeculativeTokens, uint32_t numDraftTokens, uint32_t vocabSize, int64_t const* seed,
    int64_t const* offset)
{
    uint32_t bx = blockIdx.x;
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    if (bx >= batchSize)
    {
        return;
    }

    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    using BlockScan = cub::BlockScan<float, BLOCK_SIZE>;

    __shared__ union
    {
        typename BlockReduce::TempStorage reduce;
        typename BlockScan::TempStorage scan;
    } tempStorage;

    __shared__ int32_t sLastAcceptedLocalIdx;
    __shared__ uint32_t sNumAcceptedTokens;
    __shared__ int32_t sCurIndex;
    __shared__ int32_t sFirstChild;
    __shared__ bool sHasTerminalToken;
    __shared__ bool sAcceptedSibling;
    __shared__ float sDiffSum;
    __shared__ float sTargetMass;
    __shared__ float sPrefixBase;
    __shared__ int32_t sWinnerIndex;
    __shared__ int64_t sSampledToken;

    uint32_t batchOffset = bx * numDraftTokens;

    curandStatePhilox4_32_10_t state;
    if (tid == 0)
    {
        curand_init(
            static_cast<uint64_t>(seed[0]), static_cast<uint64_t>(bx), static_cast<uint64_t>(offset[0]), &state);
    }
    __syncthreads();

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
        sCurIndex = 0;
        sHasTerminalToken = false;
    }
    __syncthreads();
    bool const hasCompactTargetSupport = targetSupportIndices != nullptr && targetSupportLengths != nullptr;

    for (uint32_t j = 1; j < numSpeculativeTokens; ++j)
    {
        // Advance to the first child of the last accepted node.
        // Continuing the example above:
        //   j = 1, curIndex = 0 (E)  -> firstChild = F1
        //   j = 2, curIndex = 1 (F1) -> firstChild = G1
        if (tid == 0)
        {
            sFirstChild = retrieveNextToken[batchOffset + sLastAcceptedLocalIdx];
            sCurIndex = sFirstChild;
            sAcceptedSibling = false;
        }
        __syncthreads();

        while (sCurIndex != -1 && !sAcceptedSibling)
        {
            if (tid == 0)
            {
                int32_t const draftLocalIdx = sCurIndex; // retrieveIndex is identity: draftLocalIdx == curIndex
                int64_t const draftTokenId = candidates[batchOffset + sCurIndex];
                int32_t const draftProbRow = draftProbIndices[batchOffset + sCurIndex];
                float const pDraft
                    = draftProbs[(static_cast<uint64_t>(bx) * numDraftProbRows + draftProbRow) * vocabSize
                        + draftTokenId];
                float const pTarget
                    = targetProbs[(static_cast<uint64_t>(bx) * numDraftTokens + sLastAcceptedLocalIdx) * vocabSize
                        + draftTokenId];

                float const acceptProb = fminf(1.0f, pTarget / (pDraft + 1e-10f));
                float const u = curand_uniform(&state);

                if (u < acceptProb)
                {
                    acceptToken[bx * numSpeculativeTokens + sNumAcceptedTokens] = draftTokenId;
                    ++sNumAcceptedTokens;
                    acceptIndex[bx * numSpeculativeTokens + sNumAcceptedTokens] = draftLocalIdx;
                    sLastAcceptedLocalIdx = draftLocalIdx;
                    sAcceptedSibling = true;
                }
                else
                {
                    sCurIndex = retrieveNextSibling[batchOffset + sCurIndex];
                }
            }
            __syncthreads();
        }

        if (sCurIndex == -1)
        {
            // All siblings exhausted. Two sub-cases:
            // (a) firstChild == -1: leaf node, no draft tokens at this depth.
            //     Emit the final bonus token sampled from q(.|lastAcceptedLocalIdx).
            // (b) firstChild != -1: every sibling was rejected -> sample correction token
            //     from relu(q - p) at firstChild's position to restore the target distribution.
            if (sFirstChild == -1)
            {
                if (tid == 0)
                {
                    float const* tProbs = targetProbs
                        + (static_cast<uint64_t>(bx) * numDraftTokens + sLastAcceptedLocalIdx) * vocabSize;
                    if (hasCompactTargetSupport)
                    {
                        uint32_t const supportOffset
                            = (static_cast<uint64_t>(bx) * numDraftTokens + sLastAcceptedLocalIdx)
                            * maxTargetSupportSize;
                        uint32_t const supportSize
                            = static_cast<uint32_t>(targetSupportLengths[batchOffset + sLastAcceptedLocalIdx]);
                        acceptToken[bx * numSpeculativeTokens + sNumAcceptedTokens] = sampleFromIndexedDistribution(
                            state, tProbs, targetSupportIndices + supportOffset, supportSize, vocabSize);
                    }
                    else
                    {
                        acceptToken[bx * numSpeculativeTokens + sNumAcceptedTokens]
                            = sampleFromDistribution(state, tProbs, vocabSize);
                    }
                    sHasTerminalToken = true;
                }
            }
            else
            {
                int32_t const draftProbRow = draftProbIndices[batchOffset + sFirstChild];
                float const* tProbs
                    = targetProbs + (static_cast<uint64_t>(bx) * numDraftTokens + sLastAcceptedLocalIdx) * vocabSize;
                float const* dProbs
                    = draftProbs + (static_cast<uint64_t>(bx) * numDraftProbRows + draftProbRow) * vocabSize;
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
                            float const r = curand_uniform(&state);
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
                    float threadDiffSum = 0.0f;
                    for (uint32_t v = static_cast<uint32_t>(tid); v < targetSupportSize; v += BLOCK_SIZE)
                    {
                        float const diff = tProbs[v] - dProbs[v];
                        if (diff > 0.0f)
                        {
                            threadDiffSum += diff;
                        }
                    }

                    float const diffSum = BlockReduce(tempStorage.reduce).Sum(threadDiffSum);
                    if (tid == 0)
                    {
                        sDiffSum = diffSum;
                        sPrefixBase = 0.0f;
                        sWinnerIndex = static_cast<int32_t>(targetSupportSize);
                        sSampledToken = static_cast<int64_t>(vocabSize) - 1;
                        if (diffSum > 1e-10f)
                        {
                            sTargetMass = curand_uniform(&state) * diffSum;
                        }
                        else
                        {
                            sSampledToken = sampleFromDistribution(state, tProbs, vocabSize);
                        }
                    }
                    __syncthreads();

                    if (sDiffSum > 1e-10f)
                    {
                        for (uint32_t tileStart = 0; tileStart < targetSupportSize; tileStart += BLOCK_SIZE)
                        {
                            float value = 0.0f;
                            uint32_t const v = tileStart + static_cast<uint32_t>(tid);
                            if (v < targetSupportSize)
                            {
                                float const diff = tProbs[v] - dProbs[v];
                                value = diff > 0.0f ? diff : 0.0f;
                            }

                            float inclusive = 0.0f;
                            float tileSum = 0.0f;
                            BlockScan(tempStorage.scan).InclusiveSum(value, inclusive, tileSum);
                            float const threshold = sTargetMass;
                            float const prefixBase = sPrefixBase;

                            if (value > 0.0f && prefixBase + inclusive >= threshold)
                            {
                                atomicMin(&sWinnerIndex, static_cast<int32_t>(v));
                            }
                            __syncthreads();

                            if (tid == 0)
                            {
                                if (sWinnerIndex < static_cast<int32_t>(targetSupportSize))
                                {
                                    sSampledToken = static_cast<int64_t>(sWinnerIndex);
                                }
                                sPrefixBase += tileSum;
                            }
                            __syncthreads();

                            if (sWinnerIndex < static_cast<int32_t>(targetSupportSize))
                            {
                                break;
                            }
                        }
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
        if (tid == 0)
        {
            float const* tProbs
                = targetProbs + (static_cast<uint64_t>(bx) * numDraftTokens + sLastAcceptedLocalIdx) * vocabSize;
            if (hasCompactTargetSupport)
            {
                uint32_t const supportOffset
                    = (static_cast<uint64_t>(bx) * numDraftTokens + sLastAcceptedLocalIdx) * maxTargetSupportSize;
                uint32_t const supportSize
                    = static_cast<uint32_t>(targetSupportLengths[batchOffset + sLastAcceptedLocalIdx]);
                acceptToken[bx * numSpeculativeTokens + sNumAcceptedTokens] = sampleFromIndexedDistribution(
                    state, tProbs, targetSupportIndices + supportOffset, supportSize, vocabSize);
            }
            else
            {
                acceptToken[bx * numSpeculativeTokens + sNumAcceptedTokens]
                    = sampleFromDistribution(state, tProbs, vocabSize);
            }
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
    int32_t const* retrieveNextSibling, SizeType32 batchSize, SizeType32 numDraftProbRows,
    SizeType32 maxTargetSupportSize, SizeType32 numDraftTokens, SizeType32 numSpecStep, SizeType32 vocabSize,
    int64_t const* seed, int64_t const* offset, cudaStream_t stream)
{
    constexpr int32_t kVerifyDynamicTreeRejectionBlockSize = 128;
    dim3 grid(batchSize);
    dim3 block(kVerifyDynamicTreeRejectionBlockSize);

    verifyDynamicTreeRejectionKernel<kVerifyDynamicTreeRejectionBlockSize><<<grid, block, 0, stream>>>(acceptIndex,
        acceptTokenNum, acceptToken, candidates, draftProbs, targetProbs, targetSupportIndices, targetSupportLengths,
        draftProbIndices, retrieveNextToken, retrieveNextSibling, batchSize, numDraftProbRows, maxTargetSupportSize,
        numSpecStep, numDraftTokens, vocabSize, seed, offset);

    sync_check_cuda_error(stream);
}

} // namespace kernels::speculative_decoding

TRTLLM_NAMESPACE_END
