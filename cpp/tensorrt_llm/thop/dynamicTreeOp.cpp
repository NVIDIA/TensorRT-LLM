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
} // namespace kernels::speculative_decoding

namespace torch_ext
{

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

//! In-place verify. Token tensors are int32; retrievePacked is [B, N, 3] int32 (index, next_token, next_sibling).
void verify_dynamic_tree_greedy_out_packed_op(th::Tensor& candidates, th::Tensor& retrievePacked,
    th::Tensor& targetPredict, th::Tensor& acceptIndex, th::Tensor& acceptTokenNum, th::Tensor& acceptToken,
    th::Tensor& treeValid, int64_t numSpecStep)
{
    TORCH_CHECK(candidates.is_cuda(), "candidates must be a CUDA tensor");
    TORCH_CHECK(retrievePacked.is_cuda(), "retrievePacked must be a CUDA tensor");
    TORCH_CHECK(targetPredict.is_cuda(), "targetPredict must be a CUDA tensor");
    TORCH_CHECK(acceptIndex.is_cuda(), "acceptIndex must be a CUDA tensor");
    TORCH_CHECK(acceptTokenNum.is_cuda(), "acceptTokenNum must be a CUDA tensor");
    TORCH_CHECK(acceptToken.is_cuda(), "acceptToken must be a CUDA tensor");
    TORCH_CHECK(treeValid.is_cuda(), "treeValid must be a CUDA tensor");
    TORCH_CHECK(candidates.dim() == 2, "candidates must be 2D tensor");
    TORCH_CHECK(retrievePacked.dim() == 3, "retrievePacked must be 3D tensor");
    TORCH_CHECK(targetPredict.dim() == 2, "targetPredict must be 2D tensor");
    TORCH_CHECK(candidates.scalar_type() == torch::kInt32, "candidates must be int32 tensor");
    TORCH_CHECK(retrievePacked.scalar_type() == torch::kInt32, "retrievePacked must be int32 tensor");
    TORCH_CHECK(targetPredict.scalar_type() == torch::kInt32, "targetPredict must be int32 tensor");

    int64_t batchSize = candidates.size(0);
    int64_t numDraftTokens = candidates.size(1);

    TORCH_CHECK(retrievePacked.size(0) == batchSize, "Batch size mismatch");
    TORCH_CHECK(retrievePacked.size(1) == numDraftTokens, "retrievePacked dim1 must match numDraftTokens");
    TORCH_CHECK(retrievePacked.size(2) == 3, "retrievePacked last dim must be 3");
    TORCH_CHECK(retrievePacked.is_contiguous(), "retrievePacked must be contiguous");
    TORCH_CHECK(candidates.is_contiguous(), "candidates must be contiguous");
    TORCH_CHECK(targetPredict.size(0) == batchSize, "Batch size mismatch");
    TORCH_CHECK(targetPredict.size(1) == numDraftTokens, "targetPredict size mismatch");
    TORCH_CHECK(targetPredict.is_contiguous(), "targetPredict must be contiguous");

    TORCH_CHECK(treeValid.dim() == 1, "treeValid must be 1D tensor");
    TORCH_CHECK(treeValid.size(0) >= batchSize, "treeValid buffer too small");
    TORCH_CHECK(treeValid.scalar_type() == torch::kBool, "treeValid must be bool tensor");
    TORCH_CHECK(treeValid.is_contiguous(), "treeValid must be contiguous");

    TORCH_CHECK(acceptIndex.scalar_type() == torch::kInt32, "acceptIndex must be int32 tensor");
    TORCH_CHECK(acceptTokenNum.scalar_type() == torch::kInt32, "acceptTokenNum must be int32 tensor");
    TORCH_CHECK(acceptToken.scalar_type() == torch::kInt32, "acceptToken must be int32 tensor");
    TORCH_CHECK(acceptIndex.size(0) >= batchSize && acceptIndex.size(1) >= numSpecStep, "acceptIndex buffer too small");
    TORCH_CHECK(acceptTokenNum.size(0) >= batchSize, "acceptTokenNum buffer too small");
    TORCH_CHECK(acceptToken.size(0) >= batchSize && acceptToken.size(1) >= numSpecStep, "acceptToken buffer too small");
    TORCH_CHECK(acceptIndex.is_contiguous(), "acceptIndex must be contiguous");
    TORCH_CHECK(acceptTokenNum.is_contiguous(), "acceptTokenNum must be contiguous");
    TORCH_CHECK(acceptToken.is_contiguous(), "acceptToken must be contiguous");

    auto stream = at::cuda::getCurrentCUDAStream(candidates.device().index());

    acceptIndex.zero_();
    acceptTokenNum.zero_();
    acceptToken.zero_();

    tk::invokeVerifyDynamicTreeGreedyPacked(acceptIndex.data_ptr<int32_t>(), acceptTokenNum.data_ptr<int32_t>(),
        acceptToken.data_ptr<int32_t>(), candidates.data_ptr<int32_t>(), retrievePacked.data_ptr<int32_t>(),
        targetPredict.data_ptr<int32_t>(), treeValid.data_ptr<bool>(), batchSize, numDraftTokens, numSpecStep, stream);
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

//! \brief Target-only rejection sampling verify op (no draft probabilities needed).
void verify_dynamic_tree_rejection_out_op(th::Tensor& draftTokens, th::Tensor& targetProbs,
    th::Tensor& retrieveNextToken, th::Tensor& retrieveNextSibling, th::Tensor& treeValid, th::Tensor& acceptIndex,
    th::Tensor& acceptTokenNum, th::Tensor& acceptToken, int64_t numSpecStep, th::Tensor& seed, th::Tensor& offset)
{
    TORCH_CHECK(draftTokens.dim() == 2, "draftTokens must be 2D tensor");
    TORCH_CHECK(targetProbs.dim() == 3, "targetProbs must be 3D tensor");
    TORCH_CHECK(retrieveNextToken.dim() == 2, "retrieveNextToken must be 2D tensor");
    TORCH_CHECK(retrieveNextSibling.dim() == 2, "retrieveNextSibling must be 2D tensor");
    TORCH_CHECK(treeValid.dim() == 1, "treeValid must be 1D tensor");
    TORCH_CHECK(draftTokens.scalar_type() == torch::kInt64, "draftTokens must be int64 tensor");
    TORCH_CHECK(targetProbs.scalar_type() == torch::kFloat32, "targetProbs must be float32 tensor");
    TORCH_CHECK(treeValid.scalar_type() == torch::kBool, "treeValid must be bool tensor");
    TORCH_CHECK(draftTokens.is_cuda(), "draftTokens must be a CUDA tensor");
    TORCH_CHECK(targetProbs.is_cuda(), "targetProbs must be a CUDA tensor");
    TORCH_CHECK(retrieveNextToken.is_cuda(), "retrieveNextToken must be a CUDA tensor");
    TORCH_CHECK(retrieveNextSibling.is_cuda(), "retrieveNextSibling must be a CUDA tensor");
    TORCH_CHECK(treeValid.is_cuda(), "treeValid must be a CUDA tensor");
    TORCH_CHECK(acceptIndex.is_cuda(), "acceptIndex must be a CUDA tensor");
    TORCH_CHECK(acceptTokenNum.is_cuda(), "acceptTokenNum must be a CUDA tensor");
    TORCH_CHECK(acceptToken.is_cuda(), "acceptToken must be a CUDA tensor");

    int64_t const batchSize = draftTokens.size(0);
    // draftTokens has shape [batchSize, N-1]; numDraftTokens is the total tree nodes N (including root).
    int64_t const numDraftTokens = draftTokens.size(1) + 1;
    int64_t const vocabSize = targetProbs.size(2);

    TORCH_CHECK(targetProbs.size(0) == batchSize, "targetProbs batch size mismatch");
    TORCH_CHECK(targetProbs.size(1) == numDraftTokens, "targetProbs numDraftTokens mismatch");
    TORCH_CHECK(retrieveNextToken.size(0) == batchSize, "retrieveNextToken batch size mismatch");
    TORCH_CHECK(retrieveNextToken.size(1) == numDraftTokens, "retrieveNextToken size mismatch");
    TORCH_CHECK(retrieveNextSibling.size(0) == batchSize, "retrieveNextSibling batch size mismatch");
    TORCH_CHECK(retrieveNextSibling.size(1) == numDraftTokens, "retrieveNextSibling size mismatch");
    TORCH_CHECK(treeValid.size(0) >= batchSize, "treeValid buffer too small");
    TORCH_CHECK(acceptIndex.scalar_type() == torch::kInt64, "acceptIndex must be int64 tensor");
    TORCH_CHECK(acceptTokenNum.scalar_type() == torch::kInt64, "acceptTokenNum must be int64 tensor");
    TORCH_CHECK(acceptToken.scalar_type() == torch::kInt64, "acceptToken must be int64 tensor");
    TORCH_CHECK(acceptIndex.size(0) >= batchSize && acceptIndex.size(1) >= numSpecStep, "acceptIndex buffer too small");
    TORCH_CHECK(acceptTokenNum.size(0) >= batchSize, "acceptTokenNum buffer too small");
    TORCH_CHECK(acceptToken.size(0) >= batchSize && acceptToken.size(1) >= numSpecStep, "acceptToken buffer too small");
    TORCH_CHECK(seed.scalar_type() == torch::kInt64 && seed.numel() >= 1 && seed.is_cuda(),
        "seed must be int64 CUDA tensor with >=1 element");
    TORCH_CHECK(offset.scalar_type() == torch::kInt64 && offset.numel() >= 1 && offset.is_cuda(),
        "offset must be int64 CUDA tensor with >=1 element");

    auto stream = at::cuda::getCurrentCUDAStream(draftTokens.device().index());

    acceptIndex.zero_();
    acceptTokenNum.zero_();
    acceptToken.zero_();

    tk::invokeVerifyDynamicTreeRejection(acceptIndex.data_ptr<int64_t>(), acceptTokenNum.data_ptr<int64_t>(),
        acceptToken.data_ptr<int64_t>(), draftTokens.data_ptr<int64_t>(), targetProbs.data_ptr<float>(),
        retrieveNextToken.data_ptr<int32_t>(), retrieveNextSibling.data_ptr<int32_t>(), treeValid.data_ptr<bool>(),
        batchSize, numDraftTokens, numSpecStep, vocabSize, seed.data_ptr<int64_t>(), offset.data_ptr<int64_t>(),
        stream);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
        "verify_dynamic_tree_greedy_out_packed_op("
        "Tensor candidates, Tensor retrievePacked, Tensor targetPredict, "
        "Tensor(a!) acceptIndex, Tensor(b!) acceptTokenNum, "
        "Tensor(c!) acceptToken, Tensor treeValid, int numSpecStep) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl(
        "verify_dynamic_tree_greedy_out_packed_op", &tensorrt_llm::torch_ext::verify_dynamic_tree_greedy_out_packed_op);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "verify_dynamic_tree_rejection_out_op("
        "Tensor draftTokens, Tensor targetProbs, "
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
        "compute_probs_from_logits_op("
        "Tensor logits, Tensor temperatures, Tensor? top_k=None, Tensor? top_p=None, "
        "bool skip_temperature=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("compute_probs_from_logits_op", &tensorrt_llm::torch_ext::compute_probs_from_logits_op);
}
