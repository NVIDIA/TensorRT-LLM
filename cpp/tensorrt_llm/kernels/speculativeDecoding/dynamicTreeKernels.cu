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

//! retrievePacked layout [bs, numDraftTokens, 3] int32 row-major:
//! [b,n,0]=retrieveIndex, [b,n,1]=retrieveNextToken, [b,n,2]=retrieveNextSibling
__global__ void verifyDynamicTreeGreedyPackedKernel(int32_t* acceptIndex, int32_t* acceptTokenNum, int32_t* acceptToken,
    int32_t const* candidates, int32_t const* retrievePacked, int32_t const* targetPredict, bool const* treeValid,
    uint32_t numSpeculativeTokens, uint32_t numDraftTokens)
{
    uint32_t bx = blockIdx.x;
    uint32_t batchOffset = bx * numDraftTokens;
    int32_t const* row = retrievePacked + static_cast<size_t>(bx) * numDraftTokens * 3;

    if (treeValid != nullptr && !treeValid[bx])
    {
        acceptTokenNum[bx] = 0;
        acceptIndex[bx * numSpeculativeTokens] = 0;
        acceptToken[bx * numSpeculativeTokens] = targetPredict[batchOffset];
        return;
    }

    int32_t lastAcceptedLocalIdx = row[0];
    acceptIndex[bx * numSpeculativeTokens] = lastAcceptedLocalIdx;
    uint32_t numAcceptedTokens = 0;
    int32_t curIndex = 0;

    acceptToken[bx * numSpeculativeTokens] = targetPredict[batchOffset + lastAcceptedLocalIdx];

    for (uint32_t j = 1; j < numSpeculativeTokens; ++j)
    {
        curIndex = row[static_cast<size_t>(curIndex) * 3 + 1];

        while (curIndex >= 0 && static_cast<uint32_t>(curIndex) < numDraftTokens)
        {
            int32_t draftLocalIdx = row[static_cast<size_t>(curIndex) * 3];
            int32_t draftTokenId = candidates[batchOffset + curIndex];
            int32_t targetTokenId = targetPredict[batchOffset + lastAcceptedLocalIdx];

            if (draftTokenId == targetTokenId)
            {
                ++numAcceptedTokens;
                acceptIndex[bx * numSpeculativeTokens + numAcceptedTokens] = draftLocalIdx;
                acceptToken[bx * numSpeculativeTokens + numAcceptedTokens] = targetPredict[batchOffset + draftLocalIdx];
                lastAcceptedLocalIdx = draftLocalIdx;
                break;
            }
            curIndex = row[static_cast<size_t>(curIndex) * 3 + 2];
        }

        if (curIndex < 0 || static_cast<uint32_t>(curIndex) >= numDraftTokens)
        {
            break;
        }
    }

    acceptTokenNum[bx] = numAcceptedTokens;
}

void invokeVerifyDynamicTreeGreedyPacked(int32_t* acceptIndex, int32_t* acceptTokenNum, int32_t* acceptToken,
    int32_t const* candidates, int32_t const* retrievePacked, int32_t const* targetPredict, bool const* treeValid,
    SizeType32 batchSize, SizeType32 numDraftTokens, SizeType32 numSpecStep, cudaStream_t stream)
{
    dim3 grid(batchSize);
    dim3 block(1);

    verifyDynamicTreeGreedyPackedKernel<<<grid, block, 0, stream>>>(acceptIndex, acceptTokenNum, acceptToken,
        candidates, retrievePacked, targetPredict, treeValid, numSpecStep, numDraftTokens);

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

// ---------------------------------------------------------------------------
// Target-only dynamic tree rejection sampling kernel
//
// Acceptance algorithm:
//   - For each depth, accumulate cumulative target probability across siblings.
//   - Accept the first sibling whose cumulative prob exceeds the random coin.
//   - When all siblings are rejected, sample a correction token from the
//     residual target mass (target prob for tokens NOT tried as siblings).
//
// This matches the mathematical guarantee of speculative sampling with the
// draft treated as a uniform empirical prior over the K candidate siblings.
// ---------------------------------------------------------------------------

// Maximum siblings we track per level for the correction step.
// Matches the maximum supported K branching factor (dynamic_tree_max_topK).
constexpr int32_t kMaxTriedPerLevel = 32;

//! \param acceptIndex         output [batchSize, numSpecStep] int64 — tree positions of accepted tokens.
//! \param acceptTokenNum      output [batchSize] int64 — # accepted draft tokens (excl. root).
//! \param acceptToken         output [batchSize, numSpecStep] int64 — accepted/correction token ids.
//! \param candidates          [batchSize, numDraftTokens] int64; col 0 = root (target sample).
//! \param targetProbs         [batchSize, numDraftTokens, vocabSize] float32; full-vocab target probs.
//! \param retrieveNextToken   [batchSize, numDraftTokens] int32 first-child pointer, -1=none.
//! \param retrieveNextSibling [batchSize, numDraftTokens] int32 next-sibling pointer, -1=none.
//! \param treeValid           [batchSize] bool; false means no valid tree exists for this request.
//! \param batchSize           batch size.
//! \param numSpeculativeTokens second dim of acceptIndex/acceptToken (= max_draft_len + 1).
//! \param numDraftTokens      total tree nodes per request (including root).
//! \param vocabSize           vocabulary size.
//! \param seed                [1] int64 on GPU. Philox RNG seed.
//! \param offset              [1] int64 on GPU. Philox RNG offset.
template <int32_t BLOCK_SIZE>
__global__ void verifyDynamicTreeRejectionKernel(int64_t* acceptIndex, int64_t* acceptTokenNum, int64_t* acceptToken,
    int64_t const* draftTokens, float const* targetProbs, int32_t const* retrieveNextToken,
    int32_t const* retrieveNextSibling, bool const* treeValid, uint32_t batchSize, uint32_t numSpeculativeTokens,
    uint32_t numDraftTokens, uint32_t vocabSize, int64_t const* seed, int64_t const* offset)
{
    uint32_t const bx = blockIdx.x;
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
    __shared__ int32_t sAccSibIdx;
    __shared__ int64_t sAccSibTok;
    __shared__ bool sAccepted;

    // Rejected siblings at the current depth (for correction sampling).
    __shared__ int32_t sTriedTokenIds[kMaxTriedPerLevel];
    __shared__ int32_t sNumTriedTokens;
    __shared__ float sProbResidual; // 1.0 - cumulative target prob of tried siblings

    uint32_t const batchOffset = bx * numDraftTokens;

    curandStatePhilox4_32_10_t state;
    if (tid == 0)
    {
        curand_init(
            static_cast<uint64_t>(seed[0]), static_cast<uint64_t>(bx), static_cast<uint64_t>(offset[0]), &state);
    }
    __syncthreads();

    // --- Helper lambdas ---

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

    // Block-parallel tile sampling: accumulate values, scan, find first >= sTargetMass.
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

    // Sample from full target distribution at tProbs.
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

    // Sample correction token from target excluding tried siblings.
    // Correction prob at token v = target_prob[v] if v was not tried, else 0.
    // probResidual = 1.0 - sum of target probs of tried siblings (pre-computed).
    auto sampleResidualWithTriedTokens = [&](float const* tProbs, float probResidual)
    {
        bool const useVectorizedLoads = canVectorizeLoad(tProbs, vocabSize);
        uint32_t const numIters = (vocabSize + BLOCK_SIZE * kVecSize - 1) / (BLOCK_SIZE * kVecSize);

        if (tid == 0)
        {
            sPrefixBase = 0.0f;
            sWinnerIndex = static_cast<int32_t>(vocabSize);
            sLastValidIndex = -1;
            sSampledToken = static_cast<int64_t>(vocabSize) - 1;
            if (probResidual > 1e-10f)
            {
                sTargetMass = curand_uniform_open_right(state) * probResidual;
            }
            else
            {
                // Nearly zero residual: fall back to argmax over full target.
                sSampledToken = sampleFromDistribution(state, tProbs, vocabSize);
            }
        }
        __syncthreads();

        if (probResidual <= 1e-10f)
        {
            return;
        }

#pragma unroll 2
        for (uint32_t i = 0; i < numIters; ++i)
        {
            uint32_t const base = (i * BLOCK_SIZE + static_cast<uint32_t>(tid)) * kVecSize;
            auto const qVec = loadProbVec(tProbs, base, useVectorizedLoads, vocabSize);
            float value[kVecSize];
#pragma unroll
            for (uint32_t j = 0; j < kVecSize; ++j)
            {
                uint32_t const v = base + j;
                if (v >= vocabSize)
                {
                    value[j] = 0.0f;
                    continue;
                }
                // Zero out tried siblings.
                bool inTried = false;
                for (int32_t k = 0; k < sNumTriedTokens; ++k)
                {
                    if (sTriedTokenIds[k] == static_cast<int32_t>(v))
                    {
                        inTried = true;
                        break;
                    }
                }
                value[j] = inTried ? 0.0f : qVec[j];
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

    // --- First-gen / dummy request: no valid tree, sample from target root. ---
    if (treeValid != nullptr && !treeValid[bx])
    {
        float const* tProbs = targetProbs + static_cast<uint64_t>(bx) * numDraftTokens * vocabSize;
        sampleTargetFullVocab(tProbs);
        if (tid == 0)
        {
            acceptIndex[bx * numSpeculativeTokens] = 0;
            acceptTokenNum[bx] = 0;
            acceptToken[bx * numSpeculativeTokens] = sSampledToken;
        }
        return;
    }

    // --- Root initialization ---
    if (tid == 0)
    {
        sLastAcceptedLocalIdx = 0;
        acceptIndex[bx * numSpeculativeTokens] = sLastAcceptedLocalIdx;
        sNumAcceptedTokens = 0;
        sHasTerminalToken = false;
    }
    __syncthreads();

    // --- Main tree traversal ---
    for (uint32_t j = 1; j < numSpeculativeTokens; ++j)
    {
        if (tid == 0)
        {
            sFirstChild = retrieveNextToken[batchOffset + sLastAcceptedLocalIdx];
        }
        __syncthreads();

        // Leaf node: emit bonus token from target at last accepted position.
        if (sFirstChild == -1)
        {
            float const* tProbs
                = targetProbs + (static_cast<uint64_t>(bx) * numDraftTokens + sLastAcceptedLocalIdx) * vocabSize;
            sampleTargetFullVocab(tProbs);
            if (tid == 0)
            {
                acceptToken[bx * numSpeculativeTokens + sNumAcceptedTokens] = sSampledToken;
                sHasTerminalToken = true;
            }
            __syncthreads();
            break;
        }

        // Try siblings using cumulative target probability.
        // Accept the first sibling whose cumulative prob exceeds the coin.
        if (tid == 0)
        {
            sNumTriedTokens = 0;
            sAccepted = false;
            float probAcc = 0.0f;
            float const coin = curand_uniform_open_right(state);

            float const* parentProbs
                = targetProbs + (static_cast<uint64_t>(bx) * numDraftTokens + sLastAcceptedLocalIdx) * vocabSize;
            int32_t childIdx = sFirstChild;
            while (childIdx != -1)
            {
                int64_t const tokenId = draftTokens[bx * (numDraftTokens - 1) + (childIdx - 1)];
                float const tProb = parentProbs[static_cast<uint64_t>(tokenId)];
                probAcc += tProb;

                if (coin <= probAcc)
                {
                    sAccSibIdx = childIdx;
                    sAccSibTok = tokenId;
                    sAccepted = true;
                    break;
                }
                else
                {
                    // Clamp the counter together with the write: sTriedTokenIds is
                    // sized [kMaxTriedPerLevel], and sNumTriedTokens bounds the
                    // read loop above. Incrementing it past the array size (when a
                    // node has more than kMaxTriedPerLevel siblings) would make that
                    // loop read out-of-bounds shared memory. Only count tokens we
                    // actually recorded; probAcc below still accumulates every
                    // sibling, so the residual normalization stays correct.
                    if (sNumTriedTokens < kMaxTriedPerLevel)
                    {
                        sTriedTokenIds[sNumTriedTokens] = static_cast<int32_t>(tokenId);
                        sNumTriedTokens = sNumTriedTokens + 1;
                    }
                    childIdx = retrieveNextSibling[batchOffset + childIdx];
                }
            }
            sProbResidual = 1.0f - probAcc;
        }
        __syncthreads();

        if (sAccepted)
        {
            if (tid == 0)
            {
                acceptToken[bx * numSpeculativeTokens + sNumAcceptedTokens] = sAccSibTok;
                ++sNumAcceptedTokens;
                acceptIndex[bx * numSpeculativeTokens + sNumAcceptedTokens] = sAccSibIdx;
                sLastAcceptedLocalIdx = sAccSibIdx;
            }
            __syncthreads();
        }
        else
        {
            // All siblings rejected: sample correction from residual target mass.
            float const* tProbs
                = targetProbs + (static_cast<uint64_t>(bx) * numDraftTokens + sLastAcceptedLocalIdx) * vocabSize;

            // Full-vocab parallel correction, excluding tried tokens.
            sampleResidualWithTriedTokens(tProbs, sProbResidual);
            if (tid == 0)
            {
                acceptToken[bx * numSpeculativeTokens + sNumAcceptedTokens] = sSampledToken;
                sHasTerminalToken = true;
            }
            __syncthreads();
            break;
        }
    }

    // Reached max speculative depth: emit bonus token from last accepted position.
    if (!sHasTerminalToken)
    {
        float const* tProbs
            = targetProbs + (static_cast<uint64_t>(bx) * numDraftTokens + sLastAcceptedLocalIdx) * vocabSize;
        sampleTargetFullVocab(tProbs);
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
    int64_t const* draftTokens, float const* targetProbs, int32_t const* retrieveNextToken,
    int32_t const* retrieveNextSibling, bool const* treeValid, SizeType32 batchSize, SizeType32 numDraftTokens,
    SizeType32 numSpecStep, SizeType32 vocabSize, int64_t const* seed, int64_t const* offset, cudaStream_t stream)
{
    constexpr int32_t kBlockSize = 256;
    dim3 grid(batchSize);
    dim3 block(kBlockSize);

    verifyDynamicTreeRejectionKernel<kBlockSize><<<grid, block, 0, stream>>>(acceptIndex, acceptTokenNum, acceptToken,
        draftTokens, targetProbs, retrieveNextToken, retrieveNextSibling, treeValid, static_cast<uint32_t>(batchSize),
        static_cast<uint32_t>(numSpecStep), static_cast<uint32_t>(numDraftTokens), static_cast<uint32_t>(vocabSize),
        seed, offset);

    sync_check_cuda_error(stream);
}

} // namespace kernels::speculative_decoding

TRTLLM_NAMESPACE_END
