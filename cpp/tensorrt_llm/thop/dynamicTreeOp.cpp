/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
namespace tl = tensorrt_llm;
namespace tk = tensorrt_llm::kernels::speculative_decoding;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! \brief Build dynamic tree structure (in-place, writes to pre-allocated output buffers)
//! All index tensors use int64 to match PyTorch's default integer dtype.
void build_dynamic_tree_op(th::Tensor& parentList, th::Tensor& selectedIndex, th::Tensor& treeMask,
    th::Tensor& positions, th::Tensor& retrieveIndex, th::Tensor& retrieveNextToken, th::Tensor& retrieveNextSibling,
    int64_t topK, int64_t depth, int64_t numDraftTokens, int64_t treeMaskMode)
{
    // Validate inputs
    TORCH_CHECK(parentList.dim() == 2, "parentList must be 2D tensor");
    TORCH_CHECK(selectedIndex.dim() == 2, "selectedIndex must be 2D tensor");
    TORCH_CHECK(parentList.scalar_type() == torch::kInt64, "parentList must be int64 tensor");
    TORCH_CHECK(selectedIndex.scalar_type() == torch::kInt64, "selectedIndex must be int64 tensor");

    int64_t batchSize = parentList.size(0);
    TORCH_CHECK(selectedIndex.size(0) == batchSize, "Batch size mismatch");
    TORCH_CHECK(selectedIndex.size(1) == numDraftTokens - 1, "selectedIndex size mismatch");

    auto device = parentList.device();
    auto stream = at::cuda::getCurrentCUDAStream(device.index());

    // Reset output buffers
    treeMask.zero_();
    positions.zero_();
    retrieveIndex.zero_();
    retrieveNextToken.fill_(-1);
    retrieveNextSibling.fill_(-1);

    // Create zero verifiedSeqLen (positions returned directly without offset)
    auto verifiedSeqLen = torch::zeros({batchSize}, torch::TensorOptions().dtype(torch::kInt32).device(device));

    // Call kernel
    tk::invokeBuildDynamicTree(parentList.data_ptr<int64_t>(), selectedIndex.data_ptr<int64_t>(),
        verifiedSeqLen.data_ptr<int32_t>(), treeMask.data_ptr(), positions.data_ptr<int32_t>(),
        retrieveIndex.data_ptr<int32_t>(), retrieveNextToken.data_ptr<int32_t>(),
        retrieveNextSibling.data_ptr<int32_t>(), batchSize, topK, depth, numDraftTokens,
        static_cast<tk::TreeMaskMode>(treeMaskMode), stream);
}

//! \brief Verify dynamic tree using greedy strategy
//! All index/token tensors use int64.
//! Returns tuple of (predicts, acceptIndex, acceptTokenNum)
std::tuple<th::Tensor, th::Tensor, th::Tensor> verify_dynamic_tree_greedy_op(th::Tensor& candidates,
    th::Tensor& retrieveIndex, th::Tensor& retrieveNextToken, th::Tensor& retrieveNextSibling,
    th::Tensor& targetPredict, int64_t numSpecStep)
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

    auto device = candidates.device();
    auto stream = at::cuda::getCurrentCUDAStream(device.index());

    // Compute total sequence length sum for predicts buffer
    int64_t seqLensSum = batchSize * numDraftTokens;

    // Allocate output tensors as int64
    auto predicts = torch::zeros({seqLensSum}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto acceptIndex
        = torch::zeros({batchSize, numSpecStep}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto acceptTokenNum = torch::zeros({batchSize}, torch::TensorOptions().dtype(torch::kInt64).device(device));

    // Convert candidates and targetPredict to int64 if needed
    auto candidatesInt64 = candidates.to(torch::kInt64);
    auto targetPredictInt64 = targetPredict.to(torch::kInt64);

    tk::invokeVerifyDynamicTreeGreedy(predicts.data_ptr<int64_t>(), acceptIndex.data_ptr<int64_t>(),
        acceptTokenNum.data_ptr<int64_t>(), candidatesInt64.data_ptr<int64_t>(), retrieveIndex.data_ptr<int32_t>(),
        retrieveNextToken.data_ptr<int32_t>(), retrieveNextSibling.data_ptr<int32_t>(),
        targetPredictInt64.data_ptr<int64_t>(), batchSize, numDraftTokens, numSpecStep, stream);

    return std::make_tuple(predicts, acceptIndex, acceptTokenNum);
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
        "int topK, int depth, int numDraftTokens, int treeMaskMode) -> ()");
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
        "Tensor retrieveNextSibling, Tensor targetPredict, int numSpecStep) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("verify_dynamic_tree_greedy_op", &tensorrt_llm::torch_ext::verify_dynamic_tree_greedy_op);
}
