/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/speculativeDecoding/dynamicTreeKernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"

namespace th = torch;
namespace tk = tensorrt_llm::kernels::speculative_decoding;

TRTLLM_NAMESPACE_BEGIN

namespace kernels::speculative_decoding
{
th::Tensor computeProbsFromLogits(th::Tensor const& logits, th::Tensor const& temperatures,
    th::optional<th::Tensor> const& topK, th::optional<th::Tensor> const& topP, bool skipTemperature,
    runtime::SizeType32 kMax);
void invokeBuildDraftProbIndices(int64_t const* topkScoreIndices, int32_t* draftProbIndices,
    runtime::SizeType32 batchSize, runtime::SizeType32 topK, runtime::SizeType32 numDraftTokens, cudaStream_t stream);
th::Tensor computeDraftProbsForDynamicTreeRejection(th::Tensor const& draftLogits, th::Tensor const& temperatures,
    runtime::SizeType32 numDraftProbRows, th::optional<th::Tensor> const& topK, th::optional<th::Tensor> const& topP,
    runtime::SizeType32 targetVocabSize, bool skipTemperature, th::optional<th::Tensor> const& d2t,
    runtime::SizeType32 kMax, bool skipAllSamplingParams);
std::tuple<th::Tensor, th::Tensor, th::Tensor> computeTargetProbsForDynamicTreeRejection(th::Tensor const& targetLogits,
    th::Tensor const& temperatures, runtime::SizeType32 numDraftTokens, th::optional<th::Tensor> const& topK,
    th::optional<th::Tensor> const& topP, bool skipTemperature, runtime::SizeType32 kMax, bool skipAllSamplingParams);
} // namespace kernels::speculative_decoding

namespace torch_ext
{

void build_draft_prob_indices_out_op(
    th::Tensor& topkScoreIndices, th::Tensor& draftProbIndices, int64_t topK, int64_t numDraftTokens)
{
    TORCH_CHECK(topkScoreIndices.is_cuda(), "topkScoreIndices must be a CUDA tensor");
    TORCH_CHECK(draftProbIndices.is_cuda(), "draftProbIndices must be a CUDA tensor");
    TORCH_CHECK(topkScoreIndices.dim() == 2, "topkScoreIndices must be a 2D tensor");
    TORCH_CHECK(draftProbIndices.dim() == 2, "draftProbIndices must be a 2D tensor");
    TORCH_CHECK(topkScoreIndices.scalar_type() == torch::kInt64, "topkScoreIndices must be int64 tensor");
    TORCH_CHECK(draftProbIndices.scalar_type() == torch::kInt32, "draftProbIndices must be int32 tensor");
    TORCH_CHECK(topkScoreIndices.size(1) == numDraftTokens, "topkScoreIndices size mismatch");
    TORCH_CHECK(draftProbIndices.size(0) == topkScoreIndices.size(0), "Batch size mismatch");
    TORCH_CHECK(draftProbIndices.size(1) == numDraftTokens + 1, "draftProbIndices size mismatch");
    TORCH_CHECK(topK > 0, "topK must be positive");
    TORCH_CHECK(numDraftTokens + 1 <= 1024, "numDraftTokens + 1 exceeds CUDA block size limit of 1024");

    auto stream = at::cuda::getCurrentCUDAStream(topkScoreIndices.device().index());
    tk::invokeBuildDraftProbIndices(topkScoreIndices.data_ptr<int64_t>(), draftProbIndices.data_ptr<int32_t>(),
        topkScoreIndices.size(0), topK, numDraftTokens, stream);
}

//! \brief Build dynamic tree structure (in-place, writes to pre-allocated output buffers)
//! All index tensors use int64 to match PyTorch's default integer dtype.
void build_dynamic_tree_op(th::Tensor& parentList, th::Tensor& selectedIndex, th::Tensor& treeMask,
    th::Tensor& positions, th::Tensor& retrieveIndex, th::Tensor& retrieveNextToken, th::Tensor& retrieveNextSibling,
    int64_t topK, int64_t depth, int64_t numDraftTokens, int64_t treeMaskMode, int64_t numInt32PerRow)
{
    // Validate inputs
    TORCH_CHECK(parentList.dim() == 2, "parentList must be 2D tensor");
    TORCH_CHECK(selectedIndex.dim() == 2, "selectedIndex must be 2D tensor");
    TORCH_CHECK(parentList.scalar_type() == torch::kInt64, "parentList must be int64 tensor");
    TORCH_CHECK(selectedIndex.scalar_type() == torch::kInt64, "selectedIndex must be int64 tensor");

    int64_t batchSize = parentList.size(0);
    TORCH_CHECK(selectedIndex.size(0) == batchSize, "Batch size mismatch");
    TORCH_CHECK(selectedIndex.size(1) == numDraftTokens - 1, "selectedIndex size mismatch");
    TORCH_CHECK(numDraftTokens <= 1024, "numDraftTokens (", numDraftTokens, ") exceeds CUDA block size limit of 1024");

    auto device = parentList.device();
    auto stream = at::cuda::getCurrentCUDAStream(device.index());

    // Reset output buffers
    treeMask.zero_();
    positions.zero_();
    retrieveIndex.zero_();
    retrieveNextToken.fill_(-1);
    retrieveNextSibling.fill_(-1);

    // Call kernel
    tk::invokeBuildDynamicTree(parentList.data_ptr<int64_t>(), selectedIndex.data_ptr<int64_t>(), treeMask.data_ptr(),
        positions.data_ptr<int32_t>(), retrieveIndex.data_ptr<int32_t>(), retrieveNextToken.data_ptr<int32_t>(),
        retrieveNextSibling.data_ptr<int32_t>(), batchSize, topK, depth, numDraftTokens,
        static_cast<tk::TreeMaskMode>(treeMaskMode), stream, numInt32PerRow);
}

//! \brief Verify dynamic tree using greedy strategy
std::tuple<th::Tensor, th::Tensor, th::Tensor, th::Tensor> verify_dynamic_tree_greedy_op(th::Tensor& candidates,
    th::Tensor& retrieveIndex, th::Tensor& retrieveNextToken, th::Tensor& retrieveNextSibling,
    th::Tensor& targetPredict, th::Tensor& treeValid, int64_t numSpecStep)
{
    // Validate inputs
    TORCH_CHECK(candidates.dim() == 2, "candidates must be 2D tensor");
    TORCH_CHECK(retrieveIndex.dim() == 2, "retrieveIndex must be 2D tensor");
    TORCH_CHECK(retrieveNextToken.dim() == 2, "retrieveNextToken must be 2D tensor");
    TORCH_CHECK(retrieveNextSibling.dim() == 2, "retrieveNextSibling must be 2D tensor");
    TORCH_CHECK(targetPredict.dim() == 2, "targetPredict must be 2D tensor");

    int64_t batchSize = candidates.size(0);
    int64_t numDraftTokens = candidates.size(1);

    TORCH_CHECK(retrieveIndex.size(0) == batchSize, "Batch size mismatch");
    TORCH_CHECK(retrieveNextToken.size(0) == batchSize, "Batch size mismatch");
    TORCH_CHECK(retrieveNextSibling.size(0) == batchSize, "Batch size mismatch");
    TORCH_CHECK(targetPredict.size(0) == batchSize, "Batch size mismatch");

    TORCH_CHECK(retrieveIndex.size(1) == numDraftTokens, "retrieveIndex size mismatch");
    TORCH_CHECK(retrieveNextToken.size(1) == numDraftTokens, "retrieveNextToken size mismatch");
    TORCH_CHECK(retrieveNextSibling.size(1) == numDraftTokens, "retrieveNextSibling size mismatch");
    TORCH_CHECK(targetPredict.size(1) == numDraftTokens, "targetPredict size mismatch");

    TORCH_CHECK(treeValid.dim() == 1, "treeValid must be 1D tensor");
    TORCH_CHECK(treeValid.size(0) >= batchSize, "treeValid buffer too small");
    TORCH_CHECK(treeValid.scalar_type() == torch::kBool, "treeValid must be bool tensor");

    auto device = candidates.device();
    auto stream = at::cuda::getCurrentCUDAStream(device.index());

    // Compute total sequence length sum for predicts buffer
    int64_t seqLensSum = batchSize * numDraftTokens;

    // Allocate output tensors as int64
    auto predicts = torch::zeros({seqLensSum}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto acceptIndex
        = torch::zeros({batchSize, numSpecStep}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto acceptTokenNum = torch::zeros({batchSize}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto acceptToken
        = torch::zeros({batchSize, numSpecStep}, torch::TensorOptions().dtype(torch::kInt64).device(device));

    // Convert candidates and targetPredict to int64 if needed
    auto candidatesInt64 = candidates.to(torch::kInt64);
    auto targetPredictInt64 = targetPredict.to(torch::kInt64);

    tk::invokeVerifyDynamicTreeGreedy(predicts.data_ptr<int64_t>(), acceptIndex.data_ptr<int64_t>(),
        acceptTokenNum.data_ptr<int64_t>(), acceptToken.data_ptr<int64_t>(), candidatesInt64.data_ptr<int64_t>(),
        retrieveIndex.data_ptr<int32_t>(), retrieveNextToken.data_ptr<int32_t>(),
        retrieveNextSibling.data_ptr<int32_t>(), targetPredictInt64.data_ptr<int64_t>(), treeValid.data_ptr<bool>(),
        batchSize, numDraftTokens, numSpecStep, stream);

    return std::make_tuple(predicts, acceptIndex, acceptTokenNum, acceptToken);
}

//! \brief In-place variant of verify_dynamic_tree_greedy_op.
//! Writes to pre-allocated output buffers instead of allocating new tensors.
void verify_dynamic_tree_greedy_out_op(th::Tensor& candidates, th::Tensor& retrieveIndex, th::Tensor& retrieveNextToken,
    th::Tensor& retrieveNextSibling, th::Tensor& targetPredict, th::Tensor& predicts, th::Tensor& acceptIndex,
    th::Tensor& acceptTokenNum, th::Tensor& acceptToken, th::Tensor& treeValid, int64_t numSpecStep)
{
    // Validate inputs
    TORCH_CHECK(candidates.dim() == 2, "candidates must be 2D tensor");
    TORCH_CHECK(retrieveIndex.dim() == 2, "retrieveIndex must be 2D tensor");
    TORCH_CHECK(retrieveNextToken.dim() == 2, "retrieveNextToken must be 2D tensor");
    TORCH_CHECK(retrieveNextSibling.dim() == 2, "retrieveNextSibling must be 2D tensor");
    TORCH_CHECK(targetPredict.dim() == 2, "targetPredict must be 2D tensor");
    TORCH_CHECK(candidates.scalar_type() == torch::kInt64, "candidates must be int64 tensor");
    TORCH_CHECK(targetPredict.scalar_type() == torch::kInt64, "targetPredict must be int64 tensor");

    int64_t batchSize = candidates.size(0);
    int64_t numDraftTokens = candidates.size(1);

    TORCH_CHECK(retrieveIndex.size(0) == batchSize, "Batch size mismatch");
    TORCH_CHECK(retrieveNextToken.size(0) == batchSize, "Batch size mismatch");
    TORCH_CHECK(retrieveNextSibling.size(0) == batchSize, "Batch size mismatch");
    TORCH_CHECK(targetPredict.size(0) == batchSize, "Batch size mismatch");

    TORCH_CHECK(retrieveIndex.size(1) == numDraftTokens, "retrieveIndex size mismatch");
    TORCH_CHECK(retrieveNextToken.size(1) == numDraftTokens, "retrieveNextToken size mismatch");
    TORCH_CHECK(retrieveNextSibling.size(1) == numDraftTokens, "retrieveNextSibling size mismatch");
    TORCH_CHECK(targetPredict.size(1) == numDraftTokens, "targetPredict size mismatch");

    TORCH_CHECK(treeValid.dim() == 1, "treeValid must be 1D tensor");
    TORCH_CHECK(treeValid.size(0) >= batchSize, "treeValid buffer too small");
    TORCH_CHECK(treeValid.scalar_type() == torch::kBool, "treeValid must be bool tensor");

    // Validate output buffers
    TORCH_CHECK(predicts.scalar_type() == torch::kInt64, "predicts must be int64 tensor");
    TORCH_CHECK(acceptIndex.scalar_type() == torch::kInt64, "acceptIndex must be int64 tensor");
    TORCH_CHECK(acceptTokenNum.scalar_type() == torch::kInt64, "acceptTokenNum must be int64 tensor");
    TORCH_CHECK(acceptToken.scalar_type() == torch::kInt64, "acceptToken must be int64 tensor");
    TORCH_CHECK(predicts.size(0) >= batchSize * numDraftTokens, "predicts buffer too small");
    TORCH_CHECK(acceptIndex.size(0) >= batchSize && acceptIndex.size(1) >= numSpecStep, "acceptIndex buffer too small");
    TORCH_CHECK(acceptTokenNum.size(0) >= batchSize, "acceptTokenNum buffer too small");
    TORCH_CHECK(acceptToken.size(0) >= batchSize && acceptToken.size(1) >= numSpecStep, "acceptToken buffer too small");

    auto stream = at::cuda::getCurrentCUDAStream(candidates.device().index());

    // Zero output buffers in-place
    predicts.zero_();
    acceptIndex.zero_();
    acceptTokenNum.zero_();
    acceptToken.zero_();

    tk::invokeVerifyDynamicTreeGreedy(predicts.data_ptr<int64_t>(), acceptIndex.data_ptr<int64_t>(),
        acceptTokenNum.data_ptr<int64_t>(), acceptToken.data_ptr<int64_t>(), candidates.data_ptr<int64_t>(),
        retrieveIndex.data_ptr<int32_t>(), retrieveNextToken.data_ptr<int32_t>(),
        retrieveNextSibling.data_ptr<int32_t>(), targetPredict.data_ptr<int64_t>(), treeValid.data_ptr<bool>(),
        batchSize, numDraftTokens, numSpecStep, stream);
}

//! \brief In-place tree rejection sampling verify op.
//! Accepts draft tokens by rejection sampling at each depth using pre-computed probabilities.
void verify_dynamic_tree_rejection_out_op(th::Tensor& candidates, th::Tensor& draftProbs, th::Tensor& targetProbs,
    th::Tensor& targetSupportIndices, th::Tensor& targetSupportLengths, th::Tensor& draftProbIndices,
    th::Tensor& retrieveNextToken, th::Tensor& retrieveNextSibling, th::Tensor& treeValid, th::Tensor& acceptIndex,
    th::Tensor& acceptTokenNum, th::Tensor& acceptToken, int64_t numSpecStep, th::Tensor& seed, th::Tensor& offset)
{
    TORCH_CHECK(candidates.dim() == 2, "candidates must be 2D tensor");
    TORCH_CHECK(draftProbs.dim() == 3, "draftProbs must be 3D tensor");
    TORCH_CHECK(targetProbs.dim() == 3, "targetProbs must be 3D tensor");
    TORCH_CHECK(targetSupportIndices.dim() == 1 || targetSupportIndices.dim() == 3,
        "targetSupportIndices must be 1D or 3D tensor");
    TORCH_CHECK(targetSupportLengths.dim() == 1 || targetSupportLengths.dim() == 2,
        "targetSupportLengths must be 1D or 2D tensor");
    TORCH_CHECK(draftProbIndices.dim() == 2, "draftProbIndices must be 2D tensor");
    TORCH_CHECK(retrieveNextToken.dim() == 2, "retrieveNextToken must be 2D tensor");
    TORCH_CHECK(retrieveNextSibling.dim() == 2, "retrieveNextSibling must be 2D tensor");
    TORCH_CHECK(treeValid.dim() == 1, "treeValid must be 1D tensor");
    TORCH_CHECK(candidates.scalar_type() == torch::kInt64, "candidates must be int64 tensor");
    TORCH_CHECK(draftProbs.scalar_type() == torch::kFloat32, "draftProbs must be float32 tensor");
    TORCH_CHECK(targetProbs.scalar_type() == torch::kFloat32, "targetProbs must be float32 tensor");
    TORCH_CHECK(targetSupportIndices.scalar_type() == torch::kInt32, "targetSupportIndices must be int32 tensor");
    TORCH_CHECK(targetSupportLengths.scalar_type() == torch::kInt32, "targetSupportLengths must be int32 tensor");
    TORCH_CHECK(draftProbIndices.scalar_type() == torch::kInt32, "draftProbIndices must be int32 tensor");
    TORCH_CHECK(treeValid.scalar_type() == torch::kBool, "treeValid must be bool tensor");
    TORCH_CHECK(candidates.is_cuda(), "candidates must be a CUDA tensor");
    TORCH_CHECK(draftProbs.is_cuda(), "draftProbs must be a CUDA tensor");
    TORCH_CHECK(targetProbs.is_cuda(), "targetProbs must be a CUDA tensor");
    TORCH_CHECK(draftProbIndices.is_cuda(), "draftProbIndices must be a CUDA tensor");
    TORCH_CHECK(retrieveNextToken.is_cuda(), "retrieveNextToken must be a CUDA tensor");
    TORCH_CHECK(retrieveNextSibling.is_cuda(), "retrieveNextSibling must be a CUDA tensor");
    TORCH_CHECK(treeValid.is_cuda(), "treeValid must be a CUDA tensor");
    TORCH_CHECK(acceptIndex.is_cuda(), "acceptIndex must be a CUDA tensor");
    TORCH_CHECK(acceptTokenNum.is_cuda(), "acceptTokenNum must be a CUDA tensor");
    TORCH_CHECK(acceptToken.is_cuda(), "acceptToken must be a CUDA tensor");

    int64_t batchSize = candidates.size(0);
    int64_t numDraftProbRows = draftProbs.size(1);
    int64_t numDraftTokens = candidates.size(1);
    int64_t vocabSize = targetProbs.size(2);
    int64_t maxTargetSupportSize = targetSupportIndices.dim() == 3 ? targetSupportIndices.size(2) : 0;

    TORCH_CHECK(draftProbs.size(0) == batchSize, "draftProbs batch size mismatch");
    TORCH_CHECK(draftProbs.size(2) == vocabSize, "draftProbs vocabSize mismatch");
    TORCH_CHECK(targetProbs.size(0) == batchSize, "targetProbs batch size mismatch");
    TORCH_CHECK(targetProbs.size(1) == numDraftTokens, "targetProbs numDraftTokens mismatch");
    if (targetSupportIndices.numel() > 0)
    {
        TORCH_CHECK(targetSupportIndices.dim() == 3, "targetSupportIndices must be 3D when non-empty");
        TORCH_CHECK(targetSupportIndices.size(0) == batchSize, "targetSupportIndices batch size mismatch");
        TORCH_CHECK(targetSupportIndices.size(1) == numDraftTokens, "targetSupportIndices numDraftTokens mismatch");
        TORCH_CHECK(targetSupportIndices.is_cuda(), "targetSupportIndices must be a CUDA tensor when non-empty");
        TORCH_CHECK(targetSupportIndices.device() == candidates.device(),
            "targetSupportIndices must be on the same device as candidates");
    }
    if (targetSupportLengths.numel() > 0)
    {
        TORCH_CHECK(targetSupportLengths.dim() == 2, "targetSupportLengths must be 2D when non-empty");
        TORCH_CHECK(targetSupportLengths.size(0) == batchSize, "targetSupportLengths batch size mismatch");
        TORCH_CHECK(targetSupportLengths.size(1) == numDraftTokens, "targetSupportLengths numDraftTokens mismatch");
        TORCH_CHECK(targetSupportLengths.is_cuda(), "targetSupportLengths must be a CUDA tensor when non-empty");
        TORCH_CHECK(targetSupportLengths.device() == candidates.device(),
            "targetSupportLengths must be on the same device as candidates");
    }
    TORCH_CHECK((targetSupportIndices.numel() == 0) == (targetSupportLengths.numel() == 0),
        "targetSupportIndices and targetSupportLengths must both be empty or both be non-empty");
    TORCH_CHECK(draftProbIndices.size(0) == batchSize, "draftProbIndices batch size mismatch");
    TORCH_CHECK(draftProbIndices.size(1) == numDraftTokens, "draftProbIndices size mismatch");
    TORCH_CHECK(retrieveNextToken.size(0) == batchSize, "retrieveNextToken batch size mismatch");
    TORCH_CHECK(retrieveNextToken.size(1) == numDraftTokens, "retrieveNextToken size mismatch");
    TORCH_CHECK(retrieveNextSibling.size(0) == batchSize, "retrieveNextSibling batch size mismatch");
    TORCH_CHECK(retrieveNextSibling.size(1) == numDraftTokens, "retrieveNextSibling size mismatch");
    TORCH_CHECK(treeValid.size(0) >= batchSize, "treeValid buffer too small");
    TORCH_CHECK(draftProbs.device() == candidates.device(), "draftProbs must be on the same device as candidates");
    TORCH_CHECK(targetProbs.device() == candidates.device(), "targetProbs must be on the same device as candidates");
    TORCH_CHECK(
        draftProbIndices.device() == candidates.device(), "draftProbIndices must be on the same device as candidates");
    TORCH_CHECK(retrieveNextToken.device() == candidates.device(),
        "retrieveNextToken must be on the same device as candidates");
    TORCH_CHECK(retrieveNextSibling.device() == candidates.device(),
        "retrieveNextSibling must be on the same device as candidates");
    TORCH_CHECK(treeValid.device() == candidates.device(), "treeValid must be on the same device as candidates");
    TORCH_CHECK(acceptIndex.scalar_type() == torch::kInt64, "acceptIndex must be int64 tensor");
    TORCH_CHECK(acceptTokenNum.scalar_type() == torch::kInt64, "acceptTokenNum must be int64 tensor");
    TORCH_CHECK(acceptToken.scalar_type() == torch::kInt64, "acceptToken must be int64 tensor");
    TORCH_CHECK(acceptIndex.size(0) >= batchSize && acceptIndex.size(1) >= numSpecStep, "acceptIndex buffer too small");
    TORCH_CHECK(acceptTokenNum.size(0) >= batchSize, "acceptTokenNum buffer too small");
    TORCH_CHECK(acceptToken.size(0) >= batchSize && acceptToken.size(1) >= numSpecStep, "acceptToken buffer too small");
    TORCH_CHECK(acceptIndex.device() == candidates.device(), "acceptIndex must be on the same device as candidates");
    TORCH_CHECK(
        acceptTokenNum.device() == candidates.device(), "acceptTokenNum must be on the same device as candidates");
    TORCH_CHECK(acceptToken.device() == candidates.device(), "acceptToken must be on the same device as candidates");
    TORCH_CHECK(seed.scalar_type() == torch::kInt64, "seed must be int64 tensor");
    TORCH_CHECK(offset.scalar_type() == torch::kInt64, "offset must be int64 tensor");
    TORCH_CHECK(seed.numel() >= 1, "seed tensor must have at least one element");
    TORCH_CHECK(offset.numel() >= 1, "offset tensor must have at least one element");
    TORCH_CHECK(seed.is_cuda(), "seed must be CUDA tensor");
    TORCH_CHECK(offset.is_cuda(), "offset must be CUDA tensor");
    TORCH_CHECK(seed.device() == candidates.device(), "seed must be on the same device as candidates");
    TORCH_CHECK(offset.device() == candidates.device(), "offset must be on the same device as candidates");

    auto stream = at::cuda::getCurrentCUDAStream(candidates.device().index());

    acceptIndex.zero_();
    acceptTokenNum.zero_();
    acceptToken.zero_();

    tk::invokeVerifyDynamicTreeRejection(acceptIndex.data_ptr<int64_t>(), acceptTokenNum.data_ptr<int64_t>(),
        acceptToken.data_ptr<int64_t>(), candidates.data_ptr<int64_t>(), draftProbs.data_ptr<float>(),
        targetProbs.data_ptr<float>(),
        targetSupportIndices.numel() > 0 ? targetSupportIndices.data_ptr<int32_t>() : nullptr,
        targetSupportLengths.numel() > 0 ? targetSupportLengths.data_ptr<int32_t>() : nullptr,
        draftProbIndices.data_ptr<int32_t>(), retrieveNextToken.data_ptr<int32_t>(),
        retrieveNextSibling.data_ptr<int32_t>(), treeValid.data_ptr<bool>(), batchSize, numDraftProbRows,
        maxTargetSupportSize, numDraftTokens, numSpecStep, vocabSize, seed.data_ptr<int64_t>(),
        offset.data_ptr<int64_t>(), stream);
}

th::Tensor compute_draft_probs_for_dynamic_tree_rejection_op(th::Tensor draftLogits, th::Tensor temperatures,
    int64_t numDraftProbRows, int64_t targetVocabSize, th::optional<th::Tensor> topK, th::optional<th::Tensor> topP,
    bool skipTemperature, th::optional<th::Tensor> d2t, int64_t topKMax, bool skipAllSamplingParams)
{
    return tk::computeDraftProbsForDynamicTreeRejection(draftLogits, temperatures, numDraftProbRows, topK, topP,
        targetVocabSize, skipTemperature, d2t, topKMax, skipAllSamplingParams);
}

std::tuple<th::Tensor, th::Tensor, th::Tensor> compute_target_probs_for_dynamic_tree_rejection_op(
    th::Tensor targetLogits, th::Tensor temperatures, int64_t numDraftTokens, th::optional<th::Tensor> topK,
    th::optional<th::Tensor> topP, bool skipTemperature, int64_t topKMax, bool skipAllSamplingParams)
{
    return tk::computeTargetProbsForDynamicTreeRejection(
        targetLogits, temperatures, numDraftTokens, topK, topP, skipTemperature, topKMax, skipAllSamplingParams);
}

th::Tensor compute_probs_from_logits_op(th::Tensor logits, th::Tensor temperatures, th::optional<th::Tensor> topK,
    th::optional<th::Tensor> topP, bool skipTemperature)
{
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
    TORCH_CHECK(temperatures.is_cuda(), "temperatures must be a CUDA tensor");
    TORCH_CHECK(logits.dim() == 2, "logits must be a 2D tensor");
    TORCH_CHECK(temperatures.dim() == 1, "temperatures must be a 1D tensor");
    TORCH_CHECK(logits.size(0) == temperatures.size(0), "logits and temperatures size mismatch");
    if (topK.has_value() && topK->defined())
    {
        TORCH_CHECK(topK->is_cuda(), "top_k must be a CUDA tensor");
    }
    if (topP.has_value() && topP->defined())
    {
        TORCH_CHECK(topP->is_cuda(), "top_p must be a CUDA tensor");
    }

    return tk::computeProbsFromLogits(logits, temperatures, topK, topP, skipTemperature, /*kMax=*/0);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

////////////////////////////////////////////////////////////////////////////////////////////////////////////

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "build_draft_prob_indices_out_op(Tensor topkScoreIndices, Tensor(a!) draftProbIndices, "
        "int topK, int numDraftTokens) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("build_draft_prob_indices_out_op", &tensorrt_llm::torch_ext::build_draft_prob_indices_out_op);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "build_dynamic_tree_op(Tensor parentList, Tensor selectedIndex, "
        "Tensor(a!) treeMask, Tensor(b!) positions, Tensor(c!) retrieveIndex, "
        "Tensor(d!) retrieveNextToken, Tensor(e!) retrieveNextSibling, "
        "int topK, int depth, int numDraftTokens, int treeMaskMode, "
        "int numInt32PerRow) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("build_dynamic_tree_op", &tensorrt_llm::torch_ext::build_dynamic_tree_op);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "verify_dynamic_tree_greedy_op(Tensor candidates, Tensor retrieveIndex, Tensor retrieveNextToken, "
        "Tensor retrieveNextSibling, Tensor targetPredict, Tensor treeValid, int numSpecStep) -> (Tensor, Tensor, "
        "Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("verify_dynamic_tree_greedy_op", &tensorrt_llm::torch_ext::verify_dynamic_tree_greedy_op);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "verify_dynamic_tree_greedy_out_op("
        "Tensor candidates, Tensor retrieveIndex, Tensor retrieveNextToken, "
        "Tensor retrieveNextSibling, Tensor targetPredict, "
        "Tensor(a!) predicts, Tensor(b!) acceptIndex, Tensor(c!) acceptTokenNum, "
        "Tensor(d!) acceptToken, Tensor treeValid, int numSpecStep) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("verify_dynamic_tree_greedy_out_op", &tensorrt_llm::torch_ext::verify_dynamic_tree_greedy_out_op);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "verify_dynamic_tree_rejection_out_op("
        "Tensor candidates, Tensor draftProbs, Tensor targetProbs, Tensor targetSupportIndices, "
        "Tensor targetSupportLengths, Tensor draftProbIndices, "
        "Tensor retrieveNextToken, Tensor retrieveNextSibling, Tensor treeValid, "
        "Tensor(a!) acceptIndex, Tensor(b!) acceptTokenNum, Tensor(c!) acceptToken, "
        "int numSpecStep, Tensor seed, Tensor offset) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("verify_dynamic_tree_rejection_out_op", &tensorrt_llm::torch_ext::verify_dynamic_tree_rejection_out_op);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "compute_draft_probs_for_dynamic_tree_rejection_op("
        "Tensor draftLogits, Tensor temperatures, int numDraftProbRows, int targetVocabSize, "
        "Tensor? top_k=None, Tensor? top_p=None, bool skip_temperature=False, Tensor? d2t=None, "
        "int top_k_max=0, bool skip_all_sampling_params=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("compute_draft_probs_for_dynamic_tree_rejection_op",
        &tensorrt_llm::torch_ext::compute_draft_probs_for_dynamic_tree_rejection_op);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "compute_target_probs_for_dynamic_tree_rejection_op("
        "Tensor targetLogits, Tensor temperatures, int numDraftTokens, "
        "Tensor? top_k=None, Tensor? top_p=None, bool skip_temperature=False, int top_k_max=0, "
        "bool skip_all_sampling_params=False) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("compute_target_probs_for_dynamic_tree_rejection_op",
        &tensorrt_llm::torch_ext::compute_target_probs_for_dynamic_tree_rejection_op);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "compute_probs_from_logits_op("
        "Tensor logits, Tensor temperatures, Tensor? top_k=None, Tensor? top_p=None, "
        "bool skip_temperature=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("compute_probs_from_logits_op", &tensorrt_llm::torch_ext::compute_probs_from_logits_op);
}
