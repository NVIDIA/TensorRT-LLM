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
    th::Tensor& retrieveNextToken, th::Tensor& retrieveNextSibling, th::Tensor& acceptIndex, th::Tensor& acceptTokenNum,
    th::Tensor& acceptToken, int64_t numSpecStep, th::Tensor& seed, th::Tensor& offset)
{
    TORCH_CHECK(candidates.dim() == 2, "candidates must be 2D tensor");
    TORCH_CHECK(draftProbs.dim() == 3, "draftProbs must be 3D tensor");
    TORCH_CHECK(targetProbs.dim() == 3, "targetProbs must be 3D tensor");
    TORCH_CHECK(retrieveNextToken.dim() == 2, "retrieveNextToken must be 2D tensor");
    TORCH_CHECK(retrieveNextSibling.dim() == 2, "retrieveNextSibling must be 2D tensor");
    TORCH_CHECK(candidates.scalar_type() == torch::kInt64, "candidates must be int64 tensor");
    TORCH_CHECK(draftProbs.scalar_type() == torch::kFloat32, "draftProbs must be float32 tensor");
    TORCH_CHECK(targetProbs.scalar_type() == torch::kFloat32, "targetProbs must be float32 tensor");

    int64_t batchSize = candidates.size(0);
    int64_t numDraftTokens = candidates.size(1);
    int64_t vocabSize = targetProbs.size(2);

    TORCH_CHECK(draftProbs.size(0) == batchSize, "draftProbs batch size mismatch");
    TORCH_CHECK(draftProbs.size(1) == numDraftTokens - 1, "draftProbs numDraftTokens-1 mismatch");
    TORCH_CHECK(draftProbs.size(2) == vocabSize, "draftProbs vocabSize mismatch");
    TORCH_CHECK(targetProbs.size(0) == batchSize, "targetProbs batch size mismatch");
    TORCH_CHECK(targetProbs.size(1) == numDraftTokens, "targetProbs numDraftTokens mismatch");
    TORCH_CHECK(retrieveNextToken.size(0) == batchSize, "retrieveNextToken batch size mismatch");
    TORCH_CHECK(retrieveNextToken.size(1) == numDraftTokens, "retrieveNextToken size mismatch");
    TORCH_CHECK(retrieveNextSibling.size(0) == batchSize, "retrieveNextSibling batch size mismatch");
    TORCH_CHECK(retrieveNextSibling.size(1) == numDraftTokens, "retrieveNextSibling size mismatch");
    TORCH_CHECK(acceptIndex.scalar_type() == torch::kInt64, "acceptIndex must be int64 tensor");
    TORCH_CHECK(acceptTokenNum.scalar_type() == torch::kInt64, "acceptTokenNum must be int64 tensor");
    TORCH_CHECK(acceptToken.scalar_type() == torch::kInt64, "acceptToken must be int64 tensor");
    TORCH_CHECK(acceptIndex.size(0) >= batchSize && acceptIndex.size(1) >= numSpecStep, "acceptIndex buffer too small");
    TORCH_CHECK(acceptTokenNum.size(0) >= batchSize, "acceptTokenNum buffer too small");
    TORCH_CHECK(acceptToken.size(0) >= batchSize && acceptToken.size(1) >= numSpecStep, "acceptToken buffer too small");
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
        targetProbs.data_ptr<float>(), retrieveNextToken.data_ptr<int32_t>(), retrieveNextSibling.data_ptr<int32_t>(),
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "verify_dynamic_tree_rejection_out_op("
        "Tensor candidates, Tensor draftProbs, Tensor targetProbs, "
        "Tensor retrieveNextToken, Tensor retrieveNextSibling, "
        "Tensor(a!) acceptIndex, Tensor(b!) acceptTokenNum, Tensor(c!) acceptToken, "
        "int numSpecStep, Tensor seed, Tensor offset) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("verify_dynamic_tree_rejection_out_op", &tensorrt_llm::torch_ext::verify_dynamic_tree_rejection_out_op);
}
