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
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/speculativeDecoding/draftTokenTreeKernels.h"
#include "tensorrt_llm/kernels/speculativeDecoding/mtpKernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"

namespace th = torch;
namespace tl = tensorrt_llm;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::tuple<th::Tensor, th::Tensor> mtp_prepare_drafter_inputs_op(th::Tensor& inputIds, th::Tensor& seqLens,
    th::Tensor& mtpPastHiddenStatesPtrs, th::Tensor& mtpPastTokensPtrs, th::Tensor& hiddenStates,
    th::Tensor& acceptedTokens, th::Tensor& numAcceptedTokens, th::Tensor& returnInputIds,
    th::Tensor& returnHiddenStates, int64_t numMTPModules, int64_t batchSize, int64_t numContextRequest,
    int64_t hiddenSize)
{
    auto dataType = hiddenStates.scalar_type();

    // Check
    auto inputIdsSizes = inputIds.sizes();
    auto hiddenStatesSizes = hiddenStates.sizes();
    TLLM_CHECK(inputIdsSizes[0] == hiddenStatesSizes[0]);

    auto seqLensSizes = seqLens.sizes();
    TLLM_CHECK(seqLensSizes[0] == batchSize);

    auto stream = at::cuda::getCurrentCUDAStream(hiddenStates.get_device());

    // Fill params
    tk::MTPPrepareDrafterInputsParam params;
    params.numMTPModules = numMTPModules;
    params.batchSize = batchSize;
    params.numContextRequest = numContextRequest;
    params.hiddenSize = hiddenSize;
    params.inputIds = reinterpret_cast<int*>(inputIds.data_ptr());
    params.seqLens = reinterpret_cast<int*>(seqLens.data_ptr());
    params.mtpPastHiddenStatesPtrs = reinterpret_cast<void**>(mtpPastHiddenStatesPtrs.data_ptr());
    params.mtpPastTokensPtrs = reinterpret_cast<int**>(mtpPastTokensPtrs.data_ptr());
    params.hiddenStates = reinterpret_cast<void*>(hiddenStates.data_ptr());
    params.acceptedTokens = reinterpret_cast<int*>(acceptedTokens.data_ptr());
    params.numAcceptedTokens = reinterpret_cast<int*>(numAcceptedTokens.data_ptr());
    params.returnInputIds = reinterpret_cast<int*>(returnInputIds.data_ptr());
    params.returnHiddenStates = reinterpret_cast<void*>(returnHiddenStates.data_ptr());

    switch (dataType)
    {
    case torch::kFloat16:
        // Handle Float16
        tk::invokeMTPPrepareDrafterInputs<half>(params, stream);
        break;
    case torch::kFloat32:
        // Handle Float32
        tk::invokeMTPPrepareDrafterInputs<float>(params, stream);
        break;
    case torch::kBFloat16:
        // Handle BFloat16
        tk::invokeMTPPrepareDrafterInputs<__nv_bfloat16>(params, stream);
        break;
    default:
        // Handle other data types
        throw std::invalid_argument("Invalid dtype, only supports float16, float32, and bfloat16");
        break;
    }

    return std::make_tuple(returnInputIds, returnHiddenStates);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::tuple<th::Tensor, th::Tensor> mtp_sampling_and_accepted_draft_tokens_op(th::Tensor& logits,
    th::Tensor& draftTokens, th::Tensor& targetTokens, int64_t numMTPModules, int64_t batchSize,
    int64_t numContextRequest, int64_t vocabSize)
{
    int const numGenerationRequest = batchSize - numContextRequest;
    auto dataType = logits.scalar_type();

    // Check
    auto logitsSizes = logits.sizes();
    TORCH_CHECK(logitsSizes.size() == 2, "logits must be a 2D Tensor");
    TLLM_CHECK(logitsSizes[0] == (numContextRequest + numGenerationRequest * (numMTPModules + 1)));

    auto draftTokensSizes = draftTokens.sizes();
    TORCH_CHECK(draftTokensSizes.size() == 1);
    TLLM_CHECK(draftTokensSizes[0] == (numGenerationRequest * numMTPModules));

    auto stream = at::cuda::getCurrentCUDAStream(logits.get_device());
    auto acceptedTokens
        = torch::ones({batchSize, numMTPModules + 1}, at::TensorOptions().dtype(torch::kInt32).device(logits.device()));
    auto numAcceptedTokens = torch::ones({batchSize}, at::TensorOptions().dtype(torch::kInt32).device(logits.device()));

    // Fill params
    tk::MTPSampleAndAcceptDraftTokensParam params;
    params.numMTPModules = numMTPModules;
    params.batchSize = batchSize;
    params.numContextRequest = numContextRequest;
    params.vocabSize = vocabSize;
    params.draftTokens = reinterpret_cast<int*>(draftTokens.data_ptr());
    params.targetTokens = reinterpret_cast<int*>(targetTokens.data_ptr());
    params.acceptedTokens = reinterpret_cast<int*>(acceptedTokens.data_ptr());
    params.numAcceptedTokens = reinterpret_cast<int*>(numAcceptedTokens.data_ptr());
    params.logits = logits.data_ptr();

    switch (dataType)
    {
    case torch::kFloat16:
        // Handle Float16
        tk::invokeMTPSampleAndAcceptDraftTokens<half>(params, stream);
        break;
    case torch::kFloat32:
        // Handle Float32
        tk::invokeMTPSampleAndAcceptDraftTokens<float>(params, stream);
        break;
    case torch::kBFloat16:
        // Handle BFloat16
        tk::invokeMTPSampleAndAcceptDraftTokens<__nv_bfloat16>(params, stream);
        break;
    default:
        // Handle other data types
        throw std::invalid_argument("Invalid dtype, only supports float16, float32, and bfloat16");
        break;
    }

    return std::make_tuple(acceptedTokens, numAcceptedTokens);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::tuple<th::Tensor, th::Tensor> mtp_update_hidden_states_op(th::Tensor& inputIds, th::Tensor& seqLens,
    th::Tensor& targetModelHiddenStates, th::Tensor& mtpPastHiddenStatesPtrs, th::Tensor& mtpPastTokensPtrs,
    th::Tensor& numAcceptedTokens, int64_t numMTPModules, int64_t batchSize, int64_t numContextRequest,
    int64_t hiddenSize)
{
    auto dataType = targetModelHiddenStates.scalar_type();

    // Check
    auto inputIdsSizes = inputIds.sizes();
    auto targetModelHiddenStatesSize = targetModelHiddenStates.sizes();
    TLLM_CHECK(inputIdsSizes[0] == targetModelHiddenStatesSize[0]);

    auto numAcceptedTokensSize = numAcceptedTokens.sizes();
    TLLM_CHECK(numAcceptedTokensSize[0] == batchSize);

    auto stream = at::cuda::getCurrentCUDAStream(targetModelHiddenStates.get_device());

    // Fill params
    tk::MTPUpdateHiddenStatesParam params;
    params.numMTPModules = numMTPModules;
    params.batchSize = batchSize;
    params.numContextRequest = numContextRequest;
    params.hiddenSize = hiddenSize;
    params.inputIds = reinterpret_cast<int*>(inputIds.data_ptr());
    params.seqLens = reinterpret_cast<int*>(seqLens.data_ptr());
    params.targetModelHiddenStates = targetModelHiddenStates.data_ptr();
    params.mtpPastHiddenStatesPtrs = reinterpret_cast<void**>(mtpPastHiddenStatesPtrs.data_ptr());
    params.mtpPastTokensPtrs = reinterpret_cast<int**>(mtpPastTokensPtrs.data_ptr());
    params.numAcceptedTokens = reinterpret_cast<int*>(numAcceptedTokens.data_ptr());

    switch (dataType)
    {
    case torch::kFloat16:
        // Handle Float16
        tk::invokeMTPUpdateHiddenStates<half>(params, stream);
        break;
    case torch::kFloat32:
        // Handle Float32
        tk::invokeMTPUpdateHiddenStates<float>(params, stream);
        break;
    case torch::kBFloat16:
        // Handle BFloat16
        tk::invokeMTPUpdateHiddenStates<__nv_bfloat16>(params, stream);
        break;
    default:
        // Handle other data types
        throw std::invalid_argument("Invalid dtype, only supports float16, float32, and bfloat16");
        break;
    }

    return std::make_tuple(mtpPastHiddenStatesPtrs, mtpPastTokensPtrs);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::tuple<th::Tensor, th::Tensor> mtp_relaxed_acceptance_op(th::Tensor& reqSlotIds, th::Tensor& topKValue,
    th::Tensor& topKIndices, th::Tensor& draftTokens, th::Tensor& mtpRelaxedDelta, th::Tensor& numAcceptedTokens,
    th::Tensor& acceptedTokens, int64_t const numMTPModules, int64_t const batchSize, int64_t const numContextRequest,
    int64_t const relaxedTopK, double const relaxedDelta, int64_t const beginThinkingTokens,
    int64_t const endThinkingTokens)
{
    auto dataType = topKValue.scalar_type();

    // Check
    auto numGenerationRequest = batchSize - numContextRequest;

    auto topKValueSizes = topKValue.sizes();
    TLLM_CHECK(topKValueSizes[0] == numGenerationRequest);
    TLLM_CHECK(topKValueSizes[1] == numMTPModules + 1);
    TLLM_CHECK(topKValueSizes[2] == relaxedTopK);

    auto draftTokensSizes = draftTokens.sizes();
    TLLM_CHECK(draftTokensSizes[0] == numGenerationRequest);

    auto numAcceptedTokensSize = numAcceptedTokens.sizes();
    TLLM_CHECK(numAcceptedTokensSize[0] == batchSize);

    auto stream = at::cuda::getCurrentCUDAStream(numAcceptedTokens.get_device());

    // Fill params
    tk::MTPRelaxedAcceptanceParam params;
    params.numMTPModules = numMTPModules;
    params.batchSize = batchSize;
    params.numContextRequest = numContextRequest;
    params.relaxedTopK = relaxedTopK;
    params.relaxedDelta = (float) relaxedDelta;
    params.beginThinkingTokens = beginThinkingTokens;
    params.endThinkingTokens = endThinkingTokens;
    params.reqSlotIds = reinterpret_cast<int*>(reqSlotIds.data_ptr());
    params.topKValue = reinterpret_cast<void*>(topKValue.data_ptr());
    params.topKIndices = reinterpret_cast<int64_t*>(topKIndices.data_ptr());
    params.draftTokens = reinterpret_cast<int*>(draftTokens.data_ptr());
    params.mtpRelaxedDelta = reinterpret_cast<float*>(mtpRelaxedDelta.data_ptr());
    params.numAcceptedTokens = reinterpret_cast<int*>(numAcceptedTokens.data_ptr());
    params.acceptedTokens = reinterpret_cast<int*>(acceptedTokens.data_ptr());

    switch (dataType)
    {
    case torch::kFloat16:
        // Handle Float16
        tk::invokeMTPRelaxedAcceptance<half>(params, stream);
        break;
    case torch::kFloat32:
        // Handle Float32
        tk::invokeMTPRelaxedAcceptance<float>(params, stream);
        break;
    case torch::kBFloat16:
        // Handle BFloat16
        tk::invokeMTPRelaxedAcceptance<__nv_bfloat16>(params, stream);
        break;
    default:
        // Handle other data types
        throw std::invalid_argument("Invalid dtype, only supports float16, float32, and bfloat16");
        break;
    }

    return std::make_tuple(acceptedTokens, numAcceptedTokens);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
void extract_real_draft_tokens_op(th::Tensor newDraftTokens, th::Tensor draftTokensBuffer,
    th::Tensor tokensGatherIdxForDrafterModel, th::Tensor topKList, th::Tensor draftTokensIndicesCumsum,
    int64_t curDraftIdx, int64_t batchSize, int64_t maxDraftLen, int64_t maxTotalDraftTokens, int64_t maxTopK)
{
    // Args:
    // curDraftIdx: int
    // batchSize: int
    // maxTotalDraftTokens: int
    // maxTopK: int
    // tokensGatherIdxForDrafterModel: Tensor, int32, indices of the draft tokens that need to be expand this layer
    //     shape: [numTokensExpandThisLayer]
    // topKList: Tensor, int32, top k value for each expandable token
    //     shape: [numTokensExpandThisLayer]
    // draftTokensIndicesCumsum: Tensor, int32, the cumulative sum of the write back indices for each draft layer
    //     shape: [maxDraftLen + 1]
    // newDraftTokens: Tensor, int64, the new draft tokens. We only need to extract this layer's tokens and write back
    // to the draftTokensBuffer
    //     shape: [batchSize, maxTotalDraftTokens + 1 if curDraftIdx > 0 else 1, maxTopK]
    // draftTokensBuffer: Tensor, int64, the buffer to store the real draft tokens
    //     shape: [maxBatchSize, maxTotalDraftTokens + 1]

    // Check the data types
    TLLM_CHECK(tokensGatherIdxForDrafterModel.scalar_type() == torch::kInt32);
    TLLM_CHECK(topKList.scalar_type() == torch::kInt32);
    TLLM_CHECK(draftTokensIndicesCumsum.scalar_type() == torch::kInt32);
    TLLM_CHECK(newDraftTokens.scalar_type() == torch::kInt64);
    TLLM_CHECK(draftTokensBuffer.scalar_type() == torch::kInt64);

    // Check the shape of 'tokensGatherIdxForDrafterModel' and 'topKList'
    auto numTokensExpandThisLayer = tokensGatherIdxForDrafterModel.size(0);
    TLLM_CHECK(numTokensExpandThisLayer > 0);
    TLLM_CHECK(topKList.size(0) == numTokensExpandThisLayer);

    // Check the shape of 'draftTokensIndicesCumsum'
    TLLM_CHECK(draftTokensIndicesCumsum.size(0) == maxDraftLen + 1);

    // Check the shape of 'newDraftTokens'
    TLLM_CHECK(newDraftTokens.size(0) == batchSize);
    if (curDraftIdx == 0)
    {
        TLLM_CHECK(newDraftTokens.size(1) == 1);
        TLLM_CHECK(newDraftTokens.size(2) == maxTopK);
    }
    else
    {
        TLLM_CHECK(newDraftTokens.size(1) == maxTotalDraftTokens + 1);
        TLLM_CHECK(newDraftTokens.size(2) == maxTopK);
    }

    // Check the shape of 'draftTokensBuffer'
    TLLM_CHECK(draftTokensBuffer.size(1) == maxTotalDraftTokens + 1);

    auto stream = at::cuda::getCurrentCUDAStream(newDraftTokens.get_device());

    // Fill params
    tk::ExtractRealDraftTokensParam params;
    params.curDraftIdx = curDraftIdx;
    params.batchSize = batchSize;
    params.maxDraftLen = maxDraftLen;
    params.maxTotalDraftTokens = maxTotalDraftTokens;
    params.maxTopK = maxTopK;
    params.numTokensExpandThisLayer = numTokensExpandThisLayer;
    params.tokensGatherIdxForDrafterModel = reinterpret_cast<int32_t*>(tokensGatherIdxForDrafterModel.data_ptr());
    params.topKList = reinterpret_cast<int32_t*>(topKList.data_ptr());
    params.draftTokensIndicesCumsum = reinterpret_cast<int32_t*>(draftTokensIndicesCumsum.data_ptr());
    params.newDraftTokens = reinterpret_cast<int64_t*>(newDraftTokens.data_ptr());
    params.draftTokensBuffer = reinterpret_cast<int64_t*>(draftTokensBuffer.data_ptr());

    tk::invokeExtractRealDraftTokens(params, stream);
}

} // end namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "mtp_prepare_drafter_inputs_op(Tensor inputIds, Tensor seqLens, Tensor "
        "mtpPastHiddenStatesPtrs, Tensor mtpPastTokensPtrs, Tensor hiddenStates, "
        "Tensor acceptedTokens, Tensor numAcceptedTokens, Tensor returnInputIds, Tensor returnHiddenStates, "
        "int numMTPModules, int batchSize, int numContextRequest,"
        "int hiddenSize) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mtp_prepare_drafter_inputs_op", &tensorrt_llm::torch_ext::mtp_prepare_drafter_inputs_op);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "mtp_sampling_and_accepted_draft_tokens_op(Tensor logits, Tensor draftTokens, Tensor "
        "targetTokens, int numMTPModules, "
        "int batchSize, int numContextRequest, int vocabSize) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mtp_sampling_and_accepted_draft_tokens_op",
        &tensorrt_llm::torch_ext::mtp_sampling_and_accepted_draft_tokens_op);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "mtp_update_hidden_states_op(Tensor inputIds, Tensor seqLens, Tensor targetModelHiddenStates, "
        "Tensor mtpPastHiddenStatesPtrs, Tensor mtpPastTokensPtrs, Tensor numAcceptedTokens, "
        "int numMTPModules, int batchSize, int numContextRequest, int hiddenSize) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mtp_update_hidden_states_op", &tensorrt_llm::torch_ext::mtp_update_hidden_states_op);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "mtp_relaxed_acceptance_op(Tensor reqSlotIds, Tensor topKValue, Tensor topKIndices, Tensor draftTokens, "
        "Tensor mtpRelaxedDelta, Tensor numAcceptedTokens, Tensor acceptedTokens, "
        "int numMTPModules, int batchSize, int numContextRequest, int relaxedTopK, "
        "float relaxedDelta, int beginThinkingTokens, int endThinkingTokens) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mtp_relaxed_acceptance_op", &tensorrt_llm::torch_ext::mtp_relaxed_acceptance_op);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "extract_real_draft_tokens_op(Tensor newDraftTokens, Tensor draftTokensBuffer, "
        "Tensor tokensGatherIdxForDrafterModel, Tensor topKList, Tensor draftTokensIndicesCumsum, "
        "int curDraftIdx, int batchSize, int maxDraftLen, int maxTotalDraftTokens, int maxTopK) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("extract_real_draft_tokens_op", &tensorrt_llm::torch_ext::extract_real_draft_tokens_op);
}
