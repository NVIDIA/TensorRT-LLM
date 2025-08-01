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

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include "tensorrt_llm/kernels/speculativeDecoding/mtpKernels.h"

namespace th = torch;
namespace tl = tensorrt_llm;
namespace tk = tensorrt_llm::kernels;

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

} // end namespace torch_ext

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
    m.impl("mtp_prepare_drafter_inputs_op", &torch_ext::mtp_prepare_drafter_inputs_op);
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
    m.impl("mtp_sampling_and_accepted_draft_tokens_op", &torch_ext::mtp_sampling_and_accepted_draft_tokens_op);
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
    m.impl("mtp_update_hidden_states_op", &torch_ext::mtp_update_hidden_states_op);
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
    m.impl("mtp_relaxed_acceptance_op", &torch_ext::mtp_relaxed_acceptance_op);
}
